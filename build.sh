#!/usr/bin/env bash
# Usage: ./build.sh build | run <command> [args...] | clean
set -euo pipefail
cd "$(dirname "$0")"

find_python() {
    for py in python3.11 "$HOME/.pyenv/versions/3.11"*/bin/python3.11 python3 python; do
        if command -v "$py" >/dev/null 2>&1 && "$py" --version 2>&1 | grep -q "3\.11"; then
            echo "$py"; return
        fi
    done
    echo "Python 3.11 not found. Install it (e.g. 'pyenv install 3.11')." >&2
    exit 1
}

case "${1:-}" in
    build)
        [ -d .venv ] || "$(find_python)" -m venv .venv
        .venv/bin/pip install -q --upgrade pip
        .venv/bin/pip install -q -r requirements.txt
        mkdir -p models
        echo "Build done."
        ;;
    run)
        shift
        [ -x .venv/bin/python ] || { echo "Run './build.sh build' first." >&2; exit 1; }
        mkdir -p models
        cd src && exec ../.venv/bin/python main.py "$@"
        ;;
    clean)
        rm -rf .venv runs models/*.pth .pytest_cache
        find . -name __pycache__ -type d -prune -exec rm -rf {} +
        echo "Cleaned."
        ;;
    *)
        echo "Usage: ./build.sh build | run <command> [args...] | clean"
        exit 1
        ;;
esac
