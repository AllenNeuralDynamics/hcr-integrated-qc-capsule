"""Tests for neuroglancer link collection in run_capsule.py."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub out heavy dependencies before importing run_capsule
# ---------------------------------------------------------------------------

def _make_package_stub(name):
    """Return a stub module that looks like a package (has __path__)."""
    mod = types.ModuleType(name)
    mod.__path__ = []
    mod.__package__ = name
    return mod


def _ensure_stub(name, is_package=False):
    if name not in sys.modules:
        mod = _make_package_stub(name) if is_package else types.ModuleType(name)
        sys.modules[name] = mod
    return sys.modules[name]


# Top-level packages must have __path__
for _pkg in ("boto3", "matplotlib", "aind_hcr_data_loader", "aind_hcr_qc"):
    _ensure_stub(_pkg, is_package=True)

# Sub-modules
for _sub in (
    "matplotlib.pyplot",
    "aind_hcr_data_loader.codeocean_utils",
    "aind_hcr_data_loader.hcr_dataset",
    "aind_hcr_data_loader.pairwise_dataset",
    "aind_hcr_qc.constants",
    "aind_hcr_qc.viz",
    "aind_hcr_qc.viz.intergrated_datasets",
    "aind_hcr_qc.viz.spectral_unmixing",
    "aind_hcr_qc.viz.cell_x_gene",
    "aind_hcr_qc.utils",
    "aind_hcr_qc.utils.s3_qc",
):
    _ensure_stub(_sub, is_package=_sub.endswith(("viz", "utils")))

# Provide the specific names imported by run_capsule / plot_configs at module level
_s3_qc = sys.modules["aind_hcr_qc.utils.s3_qc"]
_s3_qc.QC_S3_BUCKET = "aind-scratch-data"
_s3_qc.QC_S3_PREFIX = "ctl/hcr/qc"
_s3_qc.check_plot_exists = MagicMock(return_value=False)
_s3_qc.upload_plot = MagicMock()

_co = sys.modules["aind_hcr_data_loader.codeocean_utils"]
_co.MouseRecord = MagicMock
_co.attach_mouse_record_to_workstation = MagicMock(return_value=[])
_co.print_attach_results = MagicMock()

_hcr = sys.modules["aind_hcr_data_loader.hcr_dataset"]
_hcr.create_hcr_dataset_from_schema = MagicMock()

_pw = sys.modules["aind_hcr_data_loader.pairwise_dataset"]
_pw.create_pairwise_unmixing_dataset = MagicMock()

_viz = sys.modules["aind_hcr_qc.viz.intergrated_datasets"]
_viz.plot_intensity_violins = MagicMock()
_viz.plot_gene_spot_count_pairplot = MagicMock()

_spec = sys.modules["aind_hcr_qc.viz.spectral_unmixing"]
_spec.plot_channel_intensity_histograms_by_round = MagicMock()

sys.modules["matplotlib.pyplot"].close = MagicMock()
sys.modules["matplotlib.pyplot"].gcf = MagicMock()

# plot_configs imports aind_hcr_qc.constants — use MagicMock so any attribute works
sys.modules["aind_hcr_qc.constants"] = MagicMock()

# Add capsule/code to path so we can import run_capsule directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import run_capsule  # noqa: E402  (must be after stub setup)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_round_dir(tmp_path):
    """Return a temp directory pre-populated with NG JSON files."""
    d = tmp_path / "HCR_999_round1"
    d.mkdir()

    # Valid NG link files
    (d / "fused_ng.json").write_text(
        json.dumps({"ng_link": "https://neuroglancer-demo.appspot.com/#!s3://b/fused_ng.json"})
    )
    (d / "tile_subset_corrected_ng.json").write_text(
        json.dumps({"ng_link": "https://neuroglancer-demo.appspot.com/#!s3://b/tile.json"})
    )
    (d / "cc_ng_CH_488_CH_514.json").write_text(
        json.dumps({"ng_link": "https://neuroglancer-demo.appspot.com/#!s3://b/cc.json"})
    )
    (d / "multichannel_spot_annotation_ng_link.json").write_text(
        json.dumps({"ng_link": "https://neuroglancer-demo.appspot.com/#!s3://b/spots.json"})
    )

    # Non-NG JSON files that should be ignored
    (d / "acquisition.json").write_text(json.dumps({"some": "field"}))
    (d / "subject.json").write_text(json.dumps({"subject_id": "999"}))

    return d


# ---------------------------------------------------------------------------
# _collect_ng_links_for_round
# ---------------------------------------------------------------------------

class TestCollectNgLinksForRound:
    def test_collects_all_ng_files(self, tmp_round_dir):
        links = run_capsule._collect_ng_links_for_round(tmp_round_dir)
        names = [lnk["name"] for lnk in links]
        assert "fused_ng" in names
        assert "tile_subset_corrected_ng" in names
        assert "cc_ng_CH_488_CH_514" in names
        assert "multichannel_spot_annotation_ng_link" in names

    def test_ignores_non_ng_files(self, tmp_round_dir):
        links = run_capsule._collect_ng_links_for_round(tmp_round_dir)
        names = [lnk["name"] for lnk in links]
        assert "acquisition" not in names
        assert "subject" not in names

    def test_link_has_name_and_url(self, tmp_round_dir):
        links = run_capsule._collect_ng_links_for_round(tmp_round_dir)
        for lnk in links:
            assert "name" in lnk
            assert "url" in lnk
            assert lnk["url"].startswith("http")

    def test_empty_directory(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        links = run_capsule._collect_ng_links_for_round(d)
        assert links == []

    def test_nonexistent_directory(self, tmp_path):
        links = run_capsule._collect_ng_links_for_round(tmp_path / "missing")
        assert links == []

    def test_malformed_json_skipped(self, tmp_path):
        d = tmp_path / "bad"
        d.mkdir()
        (d / "broken.json").write_text("not valid json{{")
        (d / "fused_ng.json").write_text(
            json.dumps({"ng_link": "https://example.com/#!s3://b/f.json"})
        )
        links = run_capsule._collect_ng_links_for_round(d)
        assert len(links) == 1
        assert links[0]["name"] == "fused_ng"

    def test_json_without_ng_link_key_skipped(self, tmp_path):
        d = tmp_path / "no_key"
        d.mkdir()
        (d / "metadata.json").write_text(json.dumps({"version": "2.0"}))
        links = run_capsule._collect_ng_links_for_round(d)
        assert links == []


# ---------------------------------------------------------------------------
# collect_and_upload_ng_links
# ---------------------------------------------------------------------------

class TestCollectAndUploadNgLinks:
    def _make_record(self, rounds: dict) -> MagicMock:
        record = MagicMock()
        record.rounds = rounds
        return record

    @patch("run_capsule.boto3")
    def test_uploads_consolidated_json(self, mock_boto3, tmp_path, tmp_round_dir):
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3
        # Simulate 404 on head_object (file not yet on S3)
        from botocore.exceptions import ClientError
        mock_s3.head_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": ""}}, "HeadObject"
        )

        record = self._make_record({"R1": tmp_round_dir.name})
        run_capsule.collect_and_upload_ng_links(
            "999999", record, data_dir=tmp_round_dir.parent, bucket="aind-scratch-data"
        )

        mock_s3.put_object.assert_called_once()
        call_kwargs = mock_s3.put_object.call_args[1]
        assert call_kwargs["Key"].endswith("ng_links.json")
        assert call_kwargs["ContentType"] == "application/json"

        payload = json.loads(call_kwargs["Body"].decode())
        assert payload["mouse_id"] == "999999"
        assert "R1" in payload["rounds"]
        assert len(payload["rounds"]["R1"]) >= 1

    @patch("run_capsule.boto3")
    def test_skips_when_already_exists(self, mock_boto3, tmp_path, tmp_round_dir):
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3
        # head_object succeeds → file exists
        mock_s3.head_object.return_value = {"ContentLength": 42}

        record = self._make_record({"R1": tmp_round_dir.name})
        run_capsule.collect_and_upload_ng_links(
            "999999", record, data_dir=tmp_round_dir.parent, bucket="aind-scratch-data"
        )

        mock_s3.put_object.assert_not_called()

    @patch("run_capsule.boto3")
    def test_overwrite_bypasses_skip(self, mock_boto3, tmp_path, tmp_round_dir):
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        record = self._make_record({"R1": tmp_round_dir.name})
        run_capsule.collect_and_upload_ng_links(
            "999999", record, data_dir=tmp_round_dir.parent,
            bucket="aind-scratch-data", overwrite=True,
        )

        mock_s3.put_object.assert_called_once()

    @patch("run_capsule.boto3")
    def test_correct_s3_key_format(self, mock_boto3, tmp_path, tmp_round_dir):
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3
        from botocore.exceptions import ClientError
        mock_s3.head_object.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": ""}}, "HeadObject"
        )

        record = self._make_record({"R1": tmp_round_dir.name})
        run_capsule.collect_and_upload_ng_links(
            "123456", record, data_dir=tmp_round_dir.parent, bucket="my-bucket"
        )

        key = mock_s3.put_object.call_args[1]["Key"]
        assert key == "ctl/hcr/qc/123456/ng_links.json"

    @patch("run_capsule.boto3")
    def test_multiple_rounds_all_present(self, mock_boto3, tmp_path):
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3
        from botocore.exceptions import ClientError
        mock_s3.head_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": ""}}, "HeadObject"
        )

        # Create two round directories
        for rnd in ("round1", "round2"):
            d = tmp_path / rnd
            d.mkdir()
            (d / "fused_ng.json").write_text(
                json.dumps({"ng_link": f"https://example.com/#!s3://b/{rnd}.json"})
            )

        record = self._make_record({"R1": "round1", "R2": "round2"})
        run_capsule.collect_and_upload_ng_links(
            "777", record, data_dir=tmp_path, bucket="my-bucket"
        )

        payload = json.loads(mock_s3.put_object.call_args[1]["Body"].decode())
        assert set(payload["rounds"].keys()) == {"R1", "R2"}
        assert payload["rounds"]["R1"][0]["name"] == "fused_ng"
        assert payload["rounds"]["R2"][0]["name"] == "fused_ng"
