# Source-only verification image definition. Do not build, pull or deploy in Phase A.
# A separately reviewed Python base image digest is required; there is no default.
ARG APPROVED_PYTHON_BASE_IMAGE
FROM ${APPROVED_PYTHON_BASE_IMAGE}
WORKDIR /opt/kova
COPY --chown=0:0 --chmod=0444 worker/__init__.py worker/model_artifact.py worker/model_startup.py ./worker/
COPY --chown=0:0 --chmod=0444 core/__init__.py core/current_candidates.py ./core/
COPY --chown=0:0 --chmod=0444 release/__init__.py release/model_revisions.py ./release/
COPY --chown=0:0 --chmod=0444 config/core-serving.v1.json config/model-startup.v1.json ./config/
USER 65532:65532
# Ignore PYTHON* injection, user/system site customization and bytecode writes.
ENTRYPOINT ["python3", "-E", "-S", "-B", "-m", "worker.model_startup"]
CMD []
# No loader, weights, SDK, package install, network step, port, healthcheck or exec.
