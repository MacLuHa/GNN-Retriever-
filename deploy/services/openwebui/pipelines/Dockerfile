FROM ghcr.io/open-webui/pipelines:main

# Pin v2: upstream Langfuse filter uses langfuse<3; v3 SDK API differs.
RUN pip install --no-cache-dir "langfuse>=2.0.0,<3.0.0"
