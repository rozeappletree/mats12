claude mcp add jupyter \
    --scope user \
    --env JUPYTER_URL=http://localhost:8889 \
    --env JUPYTER_TOKEN=8538dcbedea9cc29e140993cf64b3d8e26b06cac3e363b3b \
    --env ALLOW_IMG_OUTPUT=true \
    -- uvx jupyter-mcp-server@latest