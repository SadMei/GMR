# Usage: source /path/to/GMR/env.sh
GMR_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
source "$GMR_REPO_ROOT/.venv/bin/activate" || return 1
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-$GMR_REPO_ROOT/../.cache/uv}"
export UV_LINK_MODE=hardlink
export UV_HTTP_TIMEOUT=600
