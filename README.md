# YadjiOs — Assistant IA Vocal Personnel

Assistant IA vocal inspiré de JARVIS (Iron Man), fonctionnant en local sur Linux avec Python 3.11–3.13.  
LLM via **Ollama**, reconnaissance vocale **faster-whisper**, synthèse **edge-tts** (avec repli local), wake word configurable.

## Fonctionnalités

- **Voix-à-voix** : wake word, STT, LLM, TTS et pipeline complet (`core/pipeline.py`)
- **12 outils** (avec agent activé) : météo, recherche web, fichiers, système, musique, notes, calendrier, heure/date, minuteur…
- **Mémoire** : court terme (historique de conversation) + long terme vectoriel (ChromaDB, optionnel)
- **API REST** + **WebSocket** streaming (FastAPI)
- **Interface web** Gradio (chat, notes, logs, configuration)
- **Démarrage automatique** via systemd (`make install-service`)

## Prérequis

- Linux (Ubuntu 22.04+ recommandé)
- Python **3.11, 3.12 ou 3.13** (3.14+ non supporté par les wheels ML)
- Microphone et haut-parleurs
- 8 Go RAM minimum (16 Go recommandé)
- [Ollama](https://ollama.com/) (installé automatiquement par `make install`)

## Installation rapide

```bash
git clone https://github.com/yadiOs-a-darel/yadjiOs-AI.git
cd yadjiOs-AI

# Standard (Ubuntu + venv)
make install

cp .env.example .env
nano .env   # Personnaliser identité, clés API, modèles…
```

### Avec Conda

Recommandé si `python3.11` n'est pas disponible via apt :

```bash
conda create -n jarvis python=3.11 -y
conda activate jarvis
make install-conda

cp .env.example .env
```

Variables utiles pour l'installation :

```bash
# Sans mise à jour apt ni service systemd
SKIP_APT=1 SKIP_SYSTEMD=1 make install

# Cible dédiée Conda
make install-conda
```

## Démarrage

```bash
# Mode vocal (défaut) — dites « hey yadjios »
make start
# ou
python main.py --mode voice

# Mode texte (sans micro, streaming console)
make start-text

# API seule (port 8000)
make start-api

# Interface Gradio seule (port 7860)
make start-ui

# Vocal + API + interface web en parallèle
python main.py --mode all

# Logs verbeux
python main.py --mode voice --debug
```

| Mode | Commande | Description |
|------|----------|-------------|
| `voice` | `make start` | Pipeline vocal complet |
| `text` | `make start-text` | Chat console (`quit` pour quitter) |
| `api` | `make start-api` | Serveur FastAPI uniquement |
| `ui` | `make start-ui` | Dashboard Gradio uniquement |
| `all` | `python main.py --mode all` | Tout en parallèle |

## Architecture

```
config/     → Configuration Pydantic (.env)
core/       → Audio, STT, TTS, LLM, wake word, pipeline vocal
memory/     → Mémoire court/long terme, notes SQLite
tools/      → Outils LangChain (météo, web, fichiers…)
agent/      → Agent YadjiOs avec tool calling
api/        → FastAPI + WebSocket
ui/         → Dashboard Gradio
scripts/    → Installation, service systemd, lancement
```

### Flux vocal

```
Micro → Wake word → STT (Whisper) → Agent/LLM → TTS → Haut-parleurs
```

## Configuration

Copiez `.env.example` vers `.env`. Variables principales :

| Variable | Défaut | Description |
|----------|--------|-------------|
| `JARVIS_NAME` | `YadjiOs` | Nom de l'assistant |
| `USER_NAME` | `Darel` | Votre prénom |
| `USER_LOCATION` | `Bangui, RCA` | Ville pour la météo |
| `OLLAMA_MODEL` | `mistral` | Modèle LLM local (tool calling recommandé) |
| `OLLAMA_FALLBACK_MODEL` | `phi3.5` | Modèle de secours |
| `AGENT_TOOLS_ENABLED` | `false` | Activer météo, web, fichiers, etc. |
| `MEMORY_SEARCH_ENABLED` | `false` | Recherche dans la mémoire vectorielle |
| `MEMORY_SAVE_ENABLED` | `false` | Enregistrement automatique en mémoire long terme |
| `WHISPER_MODEL_SIZE` | `small` | Taille STT (`tiny` → `large-v3`) |
| `WHISPER_DEVICE` | `cpu` | `cpu` ou `cuda` (GPU NVIDIA) |
| `TTS_ENGINE` | `edge` | `edge`, `coqui`, `elevenlabs`, `pyttsx3`, `local` |
| `TTS_VOICE` | `fr-FR-DeniseNeural` | Voix edge-tts |
| `SPEAK_RESPONSES` | `true` | Lire les réponses à voix haute |
| `WAKE_WORD` | `hey yadjios` | Mot-clé d'activation |
| `WAKE_WORD_PHRASES` | *(voir .env.example)* | Variantes reconnues |
| `UI_AUTO_START` | `false` | Lancer Gradio avec `make start-text` |
| `TAVILY_API_KEY` | — | Recherche web (optionnel) |
| `OPENWEATHER_API_KEY` | — | Météo (optionnel) |
| `OPENAI_API_KEY` | — | Fallback cloud si Ollama indisponible |

> **Note :** par défaut, l'agent fonctionne en **chat direct** avec Ollama (réponses rapides).  
> Pour activer les outils, définissez `AGENT_TOOLS_ENABLED=true` et utilisez un modèle compatible tool calling (ex. `mistral`).

## Outils disponibles

| Outil | Description |
|-------|-------------|
| `recherche_web` | Actualités et faits récents (Tavily / DuckDuckGo) |
| `meteo` | Météo actuelle et prévisions |
| `heure_date` | Heure, date, calcul de durées |
| `timer` | Minuteur et alarmes |
| `prendre_note` / `lire_notes` | Notes rapides (SQLite) |
| `lister_fichiers` / `lire_fichier` / `creer_fichier` | Gestion de fichiers |
| `commande_systeme` | Commandes terminal sécurisées |
| `jouer_musique` | Lecture audio (play / stop / status) |
| `calendrier` | Événements Google Calendar |

## API

Documentation interactive : `http://localhost:8000/docs`

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/api/chat` | POST | Chat texte |
| `/api/status` | GET | Statut système |
| `/api/memory/history` | GET / DELETE | Historique de conversation |
| `/ws` | WebSocket | Streaming temps réel |

Exemple :

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Quelle heure est-il ?"}'
```

## Service systemd

```bash
# Installer / mettre à jour le service utilisateur
make install-service

# Activer au démarrage
systemctl --user enable jarvis.service
systemctl --user start jarvis.service
systemctl --user status jarvis.service
```

## Tests et qualité

```bash
make test    # pytest
make lint    # ruff + mypy
make clean   # caches Python
```

## Licence

MIT — Projet personnel de Darel / I-LINGBI
