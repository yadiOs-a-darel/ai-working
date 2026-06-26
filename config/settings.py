from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # LLM
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"
    ollama_fallback_model: str = "phi3.5"
    ollama_timeout: int = 120
    ollama_num_ctx: int = 1024
    ollama_num_predict: int = 96
    ollama_keep_alive: str = "30m"
    agent_tools_enabled: bool = False
    memory_search_enabled: bool = False
    memory_save_enabled: bool = False
    ui_auto_start: bool = False
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # STT
    whisper_model_size: Literal["tiny", "base", "small", "medium", "large-v3"] = "small"
    whisper_device: Literal["cpu", "cuda"] = "cpu"
    whisper_compute_type: str = "int8"
    whisper_language: str = "fr"

    # TTS
    tts_engine: Literal["edge", "coqui", "elevenlabs", "pyttsx3", "local"] = "edge"
    tts_voice: str = "fr-FR-DeniseNeural"
    tts_local_voice: str = "fr+f2"
    tts_spd_voice_type: str = "female1"
    tts_voice_gender: Literal["female", "male"] = "female"
    tts_rate: int = 165
    speak_responses: bool = True
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""

    # Wake word
    wake_word: str = "hey yadjios"
    wake_word_phrases: str = (
        "hey yadjios,bonjour yadjios,salut yadjios,ok yadjios,"
        "hey yadji os,bonjour yadji os"
    )
    wake_word_sensitivity: float = 0.5
    wake_greeting: str = "Oui ? Que puis-je faire pour vous ?"
    startup_greeting_enabled: bool = True
    startup_greeting: str = (
        "Bonjour {user_name}, je suis {jarvis_name}. Comment puis-je vous aider ?"
    )

    # Audio
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    audio_chunk_duration: float = 0.5
    vad_silence_threshold: float = 0.5
    mic_device_index: int = -1

    # Memory
    chroma_persist_dir: str = "./data/chroma_db"
    max_conversation_history: int = 20
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Tools
    tavily_api_key: str = ""
    openweather_api_key: str = ""
    google_calendar_credentials_path: str = "./config/google_credentials.json"

    # API & UI
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    ui_port: int = 7860
    log_level: str = "INFO"
    log_file: str = "./data/logs/jarvis.log"

    # Identité
    jarvis_name: str = "YadjiOs"
    user_name: str = "Darel"
    user_location: str = "Bangui, République Centrafricaine"
    user_language: str = "fr"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
