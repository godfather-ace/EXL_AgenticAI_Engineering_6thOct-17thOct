pip install uv

uv init simple-mcp-server
cd simple-mcp-server

uv venv .venv
source .venv/bin/activate

uv add "mcp[cli]" yfinance

