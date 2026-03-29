# hash:sha256:88b854284e0a5921df72274b82aebd2046f20fe113243bf2bdeea459ffc6bb79
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:v5.0

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN --mount=type=secret,id=git-access-token \
    GIT_ACCESS_TOKEN="$(cat /run/secrets/git-access-token)" \
    pip install -U --no-cache-dir \
    aind-data-access-api[docdb]==1.9.4 \
    codeocean==0.14.0 \
    numpy==2.4.3 \
    -e git+https://github.com/AllenNeuralDynamics/aind-hcr-data-loader.git@732acd222348aa0d0f0012fa37e1d3c8c5e56d71#egg=aind-hcr-data-loader \
    -e git+https://github.com/AllenNeuralDynamics/aind-hcr-qc.git@19bbc9fe7c026b265070c2309249e3b06406e8ff#egg=aind-hcr-qc \
    -e git+https://github.com/AllenNeuralDynamics/ophys-mfish-dataset-catalog.git@13e39676959559eebdeaf68b440254996e34692e#egg=ophys-mfish-dataset-catalog

RUN code-server --extensions-dir="/.vscode/extensions" --install-extension anthropic.claude-code

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
