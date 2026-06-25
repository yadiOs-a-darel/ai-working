"""Interface Gradio JARVIS."""

import threading
import time
from pathlib import Path

import gradio as gr
import requests
from loguru import logger

from agent.jarvis_agent import get_jarvis_agent
from config.settings import settings
from core.llm import get_llm_client
from core.tts import get_tts_engine
from core.wake_word import get_wake_word_detector
from memory.notes import get_notes_manager
from memory.short_term import ShortTermMemory

API_BASE = f"http://localhost:{settings.api_port}"

_wake_word_enabled = True
_mic_level = 0.0


def chat_fn(message: str, history: list) -> str:
    if not message.strip():
        return ""
    try:
        agent = get_jarvis_agent()
        response = agent.chat(message)
        if settings.speak_responses and response.strip():
            threading.Thread(
                target=get_tts_engine().speak,
                args=(response,),
                daemon=True,
            ).start()
        return response
    except Exception as exc:
        logger.exception("Erreur chat Gradio")
        return f"Erreur : {exc}"


def get_status_html() -> str:
    try:
        llm = get_llm_client()
        info = llm.get_model_info()
        ollama_ok = info["ollama_available"]
        color = "#00ff88" if ollama_ok else "#ff4444"
        status_text = "En ligne" if ollama_ok else "Hors ligne"
        wake = get_wake_word_detector()
        wake_status = "ON" if wake.is_active and _wake_word_enabled else "OFF"
        return f"""
        <div style="padding: 10px;">
            <h3>Statut système</h3>
            <p><b>LLM :</b> {info['model']} 
               <span style="color:{color}">● {status_text}</span></p>
            <p><b>Provider :</b> {info['provider']}</p>
            <p><b>Wake word :</b> {wake_status}</p>
            <p><b>Micro :</b> niveau {_mic_level:.2f}</p>
        </div>
        """
    except Exception as exc:
        return f"<p>Erreur statut : {exc}</p>"


def toggle_wake_word(enabled: bool) -> str:
    global _wake_word_enabled
    _wake_word_enabled = enabled
    return f"Wake word : {'ON' if enabled else 'OFF'}"


def load_notes() -> str:
    notes = get_notes_manager().get_all_notes()
    if not notes:
        return "Aucune note."
    return "\n".join(
        f"#{n['id']} — {n['content'][:80]} ({n['created_at'][:10]})"
        for n in notes
    )


def clear_history_fn() -> str:
    ShortTermMemory().clear()
    return "Historique vidé."


def load_logs() -> str:
    log_path = Path(settings.log_file)
    if not log_path.exists():
        return "Aucun log disponible."
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-50:])
    except Exception as exc:
        return f"Erreur lecture logs : {exc}"


def save_config(model: str, voice: str, sensitivity: float) -> str:
    return (
        f"Configuration mise à jour (redémarrage requis) :\n"
        f"- Modèle : {model}\n"
        f"- Voix TTS : {voice}\n"
        f"- Sensibilité wake word : {sensitivity}"
    )


def create_dashboard() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="cyan").set(
        button_primary_background_fill="#00aaff",
        button_primary_background_fill_hover="#0099ee",
    )

    with gr.Blocks(title=settings.jarvis_name, theme=theme) as demo:
        gr.Markdown(f"# 🤖 {settings.jarvis_name} — Assistant IA Personnel")

        with gr.Row():
            with gr.Column(scale=3):
                gr.ChatInterface(
                    fn=chat_fn,
                    examples=["Quelle heure est-il ?", "Quel temps fait-il ?", "Prends une note : réunion demain"],
                    title="Conversation",
                )
            with gr.Column(scale=1):
                status_display = gr.HTML(value=get_status_html())
                wake_toggle = gr.Checkbox(label="Wake word actif", value=True)
                wake_status = gr.Textbox(label="État wake word", value="Wake word : ON", interactive=False)
                refresh_btn = gr.Button("Rafraîchir statut")
                refresh_btn.click(get_status_html, outputs=status_display)
                wake_toggle.change(toggle_wake_word, inputs=wake_toggle, outputs=wake_status)

        with gr.Tabs():
            with gr.Tab("Mémoire"):
                notes_display = gr.Textbox(label="Notes", lines=10, interactive=False)
                refresh_notes = gr.Button("Rafraîchir notes")
                clear_btn = gr.Button("Vider l'historique", variant="stop")
                clear_result = gr.Textbox(label="Résultat", interactive=False)
                refresh_notes.click(load_notes, outputs=notes_display)
                clear_btn.click(clear_history_fn, outputs=clear_result)

            with gr.Tab("Configuration"):
                model_input = gr.Textbox(label="Modèle Ollama", value=settings.ollama_model)
                voice_input = gr.Textbox(label="Voix TTS", value=settings.tts_voice)
                sensitivity = gr.Slider(0, 1, value=settings.wake_word_sensitivity, label="Sensibilité wake word")
                save_btn = gr.Button("Sauvegarder")
                config_result = gr.Textbox(label="Résultat", interactive=False)
                save_btn.click(
                    save_config,
                    inputs=[model_input, voice_input, sensitivity],
                    outputs=config_result,
                )

            with gr.Tab("Logs"):
                logs_display = gr.Textbox(label="Derniers logs", lines=20, interactive=False)
                refresh_logs = gr.Button("Rafraîchir logs")
                refresh_logs.click(load_logs, outputs=logs_display)

        demo.load(get_status_html, outputs=status_display)
        demo.load(load_notes, outputs=notes_display)
        demo.load(load_logs, outputs=logs_display)

    return demo


def start_ui() -> None:
    demo = create_dashboard()
    demo.launch(
        server_name=settings.api_host,
        server_port=settings.ui_port,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    start_ui()
