# hash:sha256:5ab89bd73ee5a09e148ef32286fc28986be5cd04c55802004b38c501892bfc1f
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
    -e git+https://github.com/AllenNeuralDynamics/aind-hcr-data-loader.git@1cd089defc6c3cc13bd7fff1f39126311789f948#egg=aind-hcr-data-loader \
    -e git+https://github.com/AllenNeuralDynamics/aind-hcr-qc.git@41159afcd230fef79777dfadcf5b6dabeb97471d#egg=aind-hcr-qc

RUN code-server --extensions-dir="/.vscode/extensions" --install-extension anthropic.claude-code
