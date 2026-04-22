# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
module load gcc/14.2.0
uv sync
uv sync --dev
source .venv/bin/activate