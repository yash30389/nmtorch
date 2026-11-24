FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8080 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NMTORCH_MODEL_DIR=/opt/nmtorch \
    PYTHONPATH=/

WORKDIR /app
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --requirement /tmp/requirements.txt
RUN mkdir -p /opt/nmtorch && python -c "import urllib.request; base='https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/bb17b6c3eef36d80be441ae8e5339be66e8e3b7a/'; files={'deploy.prototxt':'MobileNetSSD_deploy.prototxt','mobilenet_iter_73000.caffemodel':'mobilenet_iter_73000.caffemodel','LICENSE':'MobileNet-SSD-LICENSE'}; [urllib.request.urlretrieve(base + source, '/opt/nmtorch/' + target) for source, target in files.items()]"
COPY --chown=10001:10001 app/ /app/

FROM base AS test
COPY --chown=10001:10001 tests/ /app/tests/
USER 10001:10001
CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]

FROM base AS runtime
USER 10001:10001
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2).read()"]

CMD ["python", "server.py"]
