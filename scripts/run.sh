#!/bin/bash
# Active Conda jarvis, .venv, ou conserve l'environnement courant.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

activate_conda_jarvis() {
    if [ "${CONDA_DEFAULT_ENV:-}" = "jarvis" ]; then
        return 0
    fi
    # systemd : CONDA_PREFIX est souvent défini sans shell Conda interactif
    if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
        export PATH="${CONDA_PREFIX}/bin:${PATH}"
        return 0
    fi
    local conda_sh
    local -a conda_candidates=()
    if [ -n "${CONDA_EXE:-}" ]; then
        conda_candidates+=("${CONDA_EXE%/*}/../etc/profile.d/conda.sh")
    fi
    conda_candidates+=(
        "$HOME/anaconda3/etc/profile.d/conda.sh"
        "$HOME/miniconda3/etc/profile.d/conda.sh"
    )
    for conda_sh in "${conda_candidates[@]}"; do
        if [ -f "$conda_sh" ]; then
            # shellcheck disable=SC1090
            source "$conda_sh"
            conda activate jarvis 2>/dev/null && return 0
        fi
    done
    return 1
}

cd "$PROJECT_DIR"

if activate_conda_jarvis; then
    : # Conda jarvis actif
elif [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${PROJECT_DIR}/.venv/bin/activate"
else
    echo "⚠  Aucun env Conda 'jarvis' ni .venv — Python système utilisé."
    echo "   Recommandé : conda activate jarvis"
fi

exec "$@"
