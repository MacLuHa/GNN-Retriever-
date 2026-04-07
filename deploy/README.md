# Deploy: Open WebUI + Retriever

This folder contains a separate deployment contour for UI and retrieval layer.

## Services

- `open-webui` - chat UI.
- `openwebui-pipelines` - pipeline runtime for retrieval filter.
- `stage7-3-retriever-api` - HTTP API wrapper over `stage7_1_retriever`.

Data-layer services (`Elasticsearch`, `Qdrant`, `Neo4j`, `Ollama`) are expected to be reachable via URLs from `deploy/.env`.

## First start

1. Copy env file:
   - `cp deploy/.env.example deploy/.env`
2. Start services:
   - `docker compose -f deploy/docker-compose.yml --env-file deploy/.env up -d --build`
3. Check health:
   - `curl http://localhost:8010/health`
   - Open `http://localhost:3000`

## Open WebUI submodule

Open WebUI is mounted as a git submodule at `services/open-webui`.

- Initialize after clone:
  - `git submodule update --init --recursive`
- Update to latest tracked branch commit:
  - `git submodule update --remote --recursive`

### Switch submodule to your fork

If you want to use your fork as `origin`:

1. Set your fork URL:
   - `git -C services/open-webui remote set-url origin <YOUR_FORK_URL>`
2. Fetch and checkout your branch:
   - `git -C services/open-webui fetch origin`
   - `git -C services/open-webui checkout <YOUR_BRANCH>`
3. Commit updated submodule pointer in the main repository.

## Pipeline setup in Open WebUI

Pipeline file is in:
- `deploy/openwebui/pipelines/retriever_pipeline.py`

In Open WebUI admin panel:
1. Go to `Admin Panel -> Pipelines`.
2. Ensure provider URL points to `http://openwebui-pipelines:9099`.
3. Upload `retriever_pipeline.py`.
4. Attach the filter pipeline to your model.

## Notes

- Logs are in English by design.
- Retriever timeout and context limits are controlled by env variables in `deploy/.env`.
- Cross-encoder reranking is configured by `RERANK_*` env variables and can be enabled with `RERANK_ENABLED=true`.
- If retriever is unavailable, pipeline falls back to normal model behavior.
