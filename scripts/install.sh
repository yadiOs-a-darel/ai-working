#!/bin/bash
set -euo pipefail

echo "=== Installation JARVIS AI ==="

# Options (export avant lancement pour personnaliser) :
#   SKIP_APT=1          — ne pas toucher à apt (idéal conda)
#   SKIP_APT_UPGRADE=1  — apt install sans upgrade système
#   SKIP_OLLAMA=1       — ne pas installer Ollama
#   SKIP_SYSTEMD=1      — ne pas configurer le service systemd
#   SKIP_ML_DOWNLOAD=1  — ne pas pré-télécharger whisper/embeddings
#   REQUIRE_CONDA=1     — exiger un env Conda (make install-conda)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Espace disque : cache/temp sur $HOME si partition projet pleine ─
PROJECT_AVAIL_KB="$(df -Pk "$PROJECT_DIR" 2>/dev/null | awk 'NR==2 {print $4}')"
if [ -n "$PROJECT_AVAIL_KB" ] && [ "$PROJECT_AVAIL_KB" -lt 5242880 ]; then
    export TMPDIR="${TMPDIR:-$HOME/tmp/jarvis-pip}"
    export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/pip}"
    mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
    echo "⚠  Partition projet peu d'espace ($(df -h "$PROJECT_DIR" | awk 'NR==2 {print $4}') libre)"
    echo "   → Cache pip et fichiers temporaires : \$HOME"
fi

# ── Auto-activation Conda (make install-conda) ───────────────────────
activate_conda_jarvis() {
    if [ -n "${CONDA_PREFIX:-}" ]; then
        return 0
    fi
    local conda_sh
    for conda_sh in \
        "${CONDA_EXE%/*}/../etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "$HOME/miniconda3/etc/profile.d/conda.sh"; do
        if [ -f "$conda_sh" ]; then
            # shellcheck disable=SC1090
            source "$conda_sh"
            if conda activate jarvis 2>/dev/null; then
                echo "✓ Environnement Conda 'jarvis' activé automatiquement"
                return 0
            fi
        fi
    done
    return 1
}

if [ "${REQUIRE_CONDA:-0}" = "1" ]; then
    activate_conda_jarvis || {
        echo "❌ make install-conda nécessite l'environnement Conda 'jarvis'."
        echo "   conda create -n jarvis python=3.11 -y"
        echo "   conda activate jarvis"
        echo "   make install-conda"
        exit 1
    }
fi

# ── Détection Python 3.11–3.13 (3.14+ non supporté par les wheels ML) ─
find_python() {
    local cmd version major minor
    # Versions explicites en priorité — évite python 3.14 système
    for cmd in "${PYTHON_CMD_OVERRIDE:-}" python3.11 python3.12 python3.13 python3 python; do
        [ -z "$cmd" ] && continue
        if command -v "$cmd" &>/dev/null; then
            version="$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
            major="${version%%.*}"
            minor="${version#*.}"
            if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ] && [ "$minor" -le 13 ]; then
                echo "$cmd"
                return 0
            fi
            if [ "$major" -eq 3 ] && [ "$minor" -ge 14 ]; then
                echo "⚠  $cmd ($version) ignoré — utilisez Python 3.11–3.13 (Conda recommandé)" >&2
            fi
        fi
    done
    return 1
}

PYTHON_CMD="$(find_python)" || {
    echo "❌ Python 3.11–3.13 introuvable."
    echo "   Conda (recommandé) :"
    echo "     conda create -n jarvis python=3.11 -y"
    echo "     conda activate jarvis"
    echo "     make install-conda"
    exit 1
}

PYTHON_VERSION="$("$PYTHON_CMD" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
echo "✓ Python détecté : $PYTHON_CMD ($PYTHON_VERSION)"

# ── Conda vs venv ───────────────────────────────────────────────────
USE_CONDA=0
if [ -n "${CONDA_PREFIX:-}" ]; then
    USE_CONDA=1
    echo "✓ Environnement Conda détecté : ${CONDA_DEFAULT_ENV:-base}"
    echo "  → Les paquets seront installés dans Conda (pas de .venv)"
fi

if [ "${REQUIRE_CONDA:-0}" = "1" ] && [ "$USE_CONDA" -eq 0 ]; then
    echo "❌ REQUIRE_CONDA=1 mais aucun environnement Conda actif."
    exit 1
fi

# ── 1. Dépendances système (audio, codecs) ──────────────────────────
if [ "${SKIP_APT:-0}" != "1" ]; then
    echo ""
    echo "── Dépendances système (apt) ──"
    if [ "${SKIP_APT_UPGRADE:-0}" = "1" ]; then
        sudo apt update
    else
        sudo apt update && sudo apt upgrade -y
    fi

    sudo apt install -y \
        portaudio19-dev libsndfile1 \
        mpg123 ffmpeg \
        libavcodec-dev libavformat-dev libavutil-dev \
        libavfilter-dev libavdevice-dev libswscale-dev libswresample-dev \
        pkg-config \
        curl git build-essential \
        python3-dev python3-pip \
        2>/dev/null || sudo apt install -y \
        portaudio19-dev libsndfile1 \
        mpg123 ffmpeg \
        libavcodec-dev libavformat-dev libavutil-dev \
        libavfilter-dev libavdevice-dev libswscale-dev libswresample-dev \
        pkg-config \
        curl git build-essential
else
    echo "── apt ignoré (SKIP_APT=1) ──"
fi

# ── 2. Ollama ───────────────────────────────────────────────────────
if [ "${SKIP_OLLAMA:-0}" != "1" ]; then
    echo ""
    echo "── Ollama ──"
    if command -v ollama &>/dev/null; then
        echo "✓ Ollama déjà installé"
    else
        curl -fsSL https://ollama.com/install.sh | sh
    fi
    if command -v systemctl &>/dev/null; then
        sudo systemctl enable ollama 2>/dev/null || true
        sudo systemctl start ollama 2>/dev/null || true
    fi
    echo "── Téléchargement modèles LLM ──"
    ollama pull mistral || echo "⚠  ollama pull mistral a échoué (relancer manuellement)"
    ollama pull phi3.5 || echo "⚠  ollama pull phi3.5 a échoué (relancer manuellement)"
else
    echo "── Ollama ignoré (SKIP_OLLAMA=1) ──"
fi

# ── 3. Environnement Python ─────────────────────────────────────────
echo ""
echo "── Environnement Python ──"
if [ "$USE_CONDA" -eq 1 ]; then
    echo "Utilisation de l'environnement Conda actif"
    PYTHON_CMD="python"
else
    if [ ! -d ".venv" ]; then
        echo "Création du virtualenv .venv avec $PYTHON_CMD"
        "$PYTHON_CMD" -m venv .venv
    else
        VENV_VERSION="$(.venv/bin/python -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || echo 0)"
        if [ "$VENV_VERSION" -ge 14 ] 2>/dev/null; then
            echo "⚠  .venv utilise Python 3.14+ — recréation avec $PYTHON_CMD"
            rm -rf .venv
            "$PYTHON_CMD" -m venv .venv
        else
            echo "✓ .venv existant conservé"
        fi
    fi
    # shellcheck disable=SC1091
    source .venv/bin/activate
    PYTHON_CMD="python"
fi

# ── 4. Dépendances Python ───────────────────────────────────────────
echo ""
echo "── Paquets Python ──"
"$PYTHON_CMD" -m pip install --upgrade pip
"$PYTHON_CMD" -m pip install -r requirements.txt

# ── 5. Pré-téléchargement modèles ML ────────────────────────────────
if [ "${SKIP_ML_DOWNLOAD:-0}" != "1" ]; then
    echo ""
    echo "── Pré-téléchargement modèles ML ──"
    "$PYTHON_CMD" -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')" \
        || echo "⚠  Téléchargement Whisper reporté au premier lancement"
    "$PYTHON_CMD" -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
        || echo "⚠  Téléchargement embeddings reporté au premier lancement"
fi

# ── 6. Dossiers data ────────────────────────────────────────────────
mkdir -p data/chroma_db data/logs

# ── 7. Fichier .env ─────────────────────────────────────────────────
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "⚠  Édite .env avec tes clés API : nano .env"
else
    echo "✓ .env existant conservé"
fi

# ── 8. Service systemd (optionnel) ──────────────────────────────────
if [ "${SKIP_SYSTEMD:-0}" != "1" ] && command -v systemctl &>/dev/null; then
    echo ""
    echo "── Service systemd ──"
    CURRENT_USER="$(whoami)"
    PYTHON_PATH="$(command -v "$PYTHON_CMD")"

    if [ "$USE_CONDA" -eq 1 ]; then
        EXEC_START="${PYTHON_PATH} ${PROJECT_DIR}/main.py --mode all"
        ENV_LINE="Environment=CONDA_PREFIX=${CONDA_PREFIX}"
    elif [ -f "${PROJECT_DIR}/.venv/bin/python" ]; then
        EXEC_START="${PROJECT_DIR}/.venv/bin/python ${PROJECT_DIR}/main.py --mode all"
        ENV_LINE=""
    else
        EXEC_START="${PYTHON_PATH} ${PROJECT_DIR}/main.py --mode all"
        ENV_LINE=""
    fi

    sudo tee /etc/systemd/system/jarvis.service > /dev/null <<EOF
[Unit]
Description=JARVIS AI Assistant
After=network.target ollama.service
Wants=ollama.service

[Service]
Type=simple
User=${CURRENT_USER}
WorkingDirectory=${PROJECT_DIR}
ExecStart=${EXEC_START}
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
Environment=PYTHONUNBUFFERED=1
${ENV_LINE}

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable jarvis 2>/dev/null || true
    echo "✓ Service jarvis.service installé"
else
    echo "── systemd ignoré ──"
fi

echo ""
echo "✅ Installation terminée !"
echo ""
if [ "$USE_CONDA" -eq 1 ]; then
    echo "   Environnement : Conda (${CONDA_DEFAULT_ENV:-base})"
    echo "   Démarrer      : conda activate jarvis && make start"
else
    echo "   Démarrer      : make start  (active .venv automatiquement)"
fi
echo "   Mode texte    : make start-text"
echo "   Interface     : http://localhost:7860"
