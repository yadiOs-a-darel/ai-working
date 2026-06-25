# JARVIS — Assistant IA Vocal Personnel

Assistant IA vocal inspiré de JARVIS (Iron Man), fonctionnant entièrement en local sur Ubuntu Linux avec Python 3.11+.

## Fonctionnalités

- **Voix-à-voix** : wake word, STT (faster-whisper), LLM (Ollama), TTS (edge-tts)
- **11 outils** : météo, recherche web, fichiers, système, musique, notes, calendrier...
- **Mémoire** : court terme (conversation) + long terme (ChromaDB vectoriel)
- **API REST** + **WebSocket** streaming (FastAPI, port 8000)
- **Interface web** Gradio (http://localhost:7860)
- **Démarrage automatique** via systemd

## Prérequis

- Ubuntu Linux 22.04+
- Python 3.11+
- Microphone et haut-parleurs
- 8 Go RAM minimum (16 Go recommandé)
- Ollama (installé automatiquement)

## Installation rapide

```bash
git clone https://github.com/ton-compte/jarvis.git
cd jarvis

# Standard (Ubuntu + venv)
make install

# Avec Conda (recommandé si python3.11 apt indisponible)
conda create -n jarvis python=3.11 -y
conda activate jarvis
make install-conda

cp .env.example .env
nano .env  # Configurer vos clés API
```

### Conda — variables utiles

```bash
# Installation sans upgrade système ni systemd
SKIP_APT=1 SKIP_SYSTEMD=1 make install

# Ou via la cible dédiée
make install-conda
```

## Démarrage

```bash
# Mode texte (test sans micro)
make start-text

# Mode vocal complet
make start

# API seule
make start-api

# Interface Gradio seule
make start-ui

# Tout en parallèle
python main.py --mode all
```

## Architecture

```
config/     → Configuration Pydantic (.env)
core/       → Audio, STT, TTS, LLM, pipeline vocal
memory/     → Mémoire court/long terme, notes SQLite
tools/      → Outils LangChain (météo, web, fichiers...)
agent/      → Agent JARVIS avec tool calling
api/        → FastAPI + WebSocket
ui/         → Dashboard Gradio
```

## Configuration

Copiez `.env.example` vers `.env` et configurez :

| Variable | Description |
|----------|-------------|
| `OLLAMA_MODEL` | Modèle LLM local (mistral, phi3.5...) |
| `WHISPER_MODEL_SIZE` | Taille modèle STT (small recommandé) |
| `TTS_VOICE` | Voix edge-tts (fr-FR-HenriNeural) |
| `TAVILY_API_KEY` | Clé API recherche web (optionnel) |

## API

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/api/chat` | POST | Chat texte |
| `/api/status` | GET | Statut système |
| `/api/memory/history` | GET/DELETE | Historique conversation |
| `/ws` | WebSocket | Streaming temps réel |

## Tests

```bash
make test
make lint
```

## Licence

MIT — Projet personnel de Darel / I-LINGBI
# yadjiOs-AI
