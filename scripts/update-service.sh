#!/bin/bash
# Installe le service systemd utilisateur (démarrage auto à la connexion).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CURRENT_USER="$(whoami)"
USER_UID="$(id -u "${CURRENT_USER}")"

CONDA_PREFIX="${CONDA_PREFIX:-$HOME/anaconda3/envs/jarvis}"
if [ ! -x "${CONDA_PREFIX}/bin/python" ]; then
    echo "❌ Python Conda introuvable : ${CONDA_PREFIX}/bin/python"
    echo "   conda activate jarvis puis relancez : make install-service"
    exit 1
fi

USER_UNIT_DIR="${HOME}/.config/systemd/user"
mkdir -p "${USER_UNIT_DIR}"

tee "${USER_UNIT_DIR}/jarvis.service" > /dev/null <<EOF
[Unit]
Description=YadjiOs AI Assistant
After=network-online.target sound.target ollama.service
Wants=network-online.target ollama.service

[Service]
Type=simple
WorkingDirectory=${PROJECT_DIR}
ExecStart=/bin/bash ${PROJECT_DIR}/scripts/run.sh python ${PROJECT_DIR}/main.py --mode voice
Restart=on-failure
RestartSec=15
StandardOutput=journal
StandardError=journal
Environment=PYTHONUNBUFFERED=1
Environment=HOME=${HOME}
Environment=XDG_RUNTIME_DIR=/run/user/${USER_UID}
Environment=PULSE_SERVER=unix:/run/user/${USER_UID}/pulse/native
Environment=PATH=${CONDA_PREFIX}/bin:/usr/local/bin:/usr/bin:/bin
Environment=CONDA_PREFIX=${CONDA_PREFIX}

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable jarvis.service

if systemctl --user is-active jarvis.service &>/dev/null; then
    systemctl --user restart jarvis.service
else
    systemctl --user start jarvis.service || true
fi

# Permet au service utilisateur de démarrer sans session graphique ouverte
if command -v loginctl &>/dev/null; then
    loginctl enable-linger "${CURRENT_USER}" 2>/dev/null || true
fi

# Ollama au démarrage machine (service système)
if command -v systemctl &>/dev/null && systemctl list-unit-files ollama.service &>/dev/null; then
    sudo systemctl enable ollama 2>/dev/null || true
fi

echo ""
echo "✓ Service utilisateur installé et activé au démarrage."
echo "  Démarrer   : systemctl --user start jarvis"
echo "  Arrêter    : systemctl --user stop jarvis"
echo "  Statut     : systemctl --user status jarvis"
echo "  Logs       : journalctl --user -u jarvis -f"
echo ""
systemctl --user status jarvis.service --no-pager | head -14 || true
