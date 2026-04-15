# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv sync
uv sync --dev
source .venv/bin/activate
module load gcc/14.2.0