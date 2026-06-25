"""Point d'entrée principal JARVIS."""

import os
import sys
import threading
from pathlib import Path

# Désactive télémétrie cloud (peut bloquer sans Internet)
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import click
import uvicorn
from loguru import logger
from rich.console import Console

from config.settings import settings

console = Console()


def setup_logging(debug: bool = False) -> None:
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()
    level = "DEBUG" if debug else settings.log_level
    log_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{line} | {message}"
    )
    logger.add(sys.stderr, level=level, format=log_format)
    logger.add(
        str(log_path),
        level=level,
        format=log_format,
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8",
    )


def print_banner() -> None:
    banner = f"""
╔═══════════════════════════════════════════╗
║         {settings.jarvis_name} v1.0.0              ║
║    Assistant IA Personnel de {settings.user_name:<10} ║
╚═══════════════════════════════════════════╝
"""
    console.print(banner, style="bold cyan")


def print_startup_status() -> None:
    from core.llm import get_llm_client
    from core.stt import get_stt_engine
    from core.tts import get_tts_engine
    from core.wake_word import get_wake_word_detector
    from memory.long_term import get_long_term_memory
    from tools.registry import get_all_tools

    try:
        stt = get_stt_engine()
        console.print(
            f"[green][✓][/green] Modèle STT chargé : faster-whisper "
            f"{settings.whisper_model_size} ({settings.whisper_device})"
        )
    except Exception as exc:
        console.print(f"[red][✗][/red] STT : {exc}")

    llm = get_llm_client()
    info = llm.get_model_info()
    provider = "Ollama" if info["ollama_available"] else info["provider"]
    console.print(f"[green][✓][/green] LLM connecté : {provider} - {info['model']}")
    if info["ollama_available"] and not llm.model_supports_tools():
        console.print(
            "[yellow][!][/yellow] Outils désactivés : ce modèle ne supporte pas le tool calling "
            "(utilisez mistral ou un modèle compatible pour les outils)"
        )
    elif not settings.agent_tools_enabled:
        console.print(
            "[yellow][!][/yellow] Agent outils désactivé — chat direct Ollama "
            "(AGENT_TOOLS_ENABLED=true pour météo, web, etc.)"
        )

    try:
        ltm = get_long_term_memory()
        console.print(
            f"[green][✓][/green] Mémoire vectorielle : ChromaDB "
            f"({ltm.count()} souvenirs)"
        )
    except Exception as exc:
        console.print(f"[yellow][!][/yellow] Mémoire long terme : {exc}")

    tools = get_all_tools()
    console.print(f"[green][✓][/green] Tools chargés : {len(tools)} outils disponibles")

    tts = get_tts_engine()
    speak_label = "activée" if settings.speak_responses else "désactivée"
    console.print(
        f"[green][✓][/green] TTS prêt : {settings.tts_engine}-tts "
        f"({settings.tts_voice}) — lecture {speak_label}"
    )

    wake = get_wake_word_detector()
    if wake.is_active:
        console.print(f'[green][✓][/green] Wake word actif : "{settings.wake_word}"')
    else:
        console.print("[yellow][!][/yellow] Wake word : mode manuel (Entrée)")

    console.print(f"[green][✓][/green] Interface web : http://localhost:{settings.ui_port}")
    console.print("\n[bold]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold]")
    console.print(
        f"  [bold green]{settings.jarvis_name} en ligne. En attente...[/bold green]"
    )
    console.print("[bold]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold]\n")


def run_voice(debug: bool) -> None:
    from core.pipeline import get_pipeline

    pipeline = get_pipeline(debug=debug)
    pipeline.run()


def run_text(debug: bool) -> None:
    from core.pipeline import get_pipeline

    pipeline = get_pipeline(debug=debug)
    console.print("[cyan]Mode texte — tapez 'quit' pour quitter[/cyan]\n")
    while True:
        try:
            text = input("[Vous]: ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue
            with console.status("[dim]Réflexion en cours...[/dim]", spinner="dots"):
                response = pipeline.process_text(
                    text, vocal_mode=False, speak=False
                )
            console.print(f"[green][{settings.jarvis_name}]:[/green] {response}\n")
        except (KeyboardInterrupt, EOFError):
            break


def run_api(debug: bool) -> None:
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=debug,
        log_level="debug" if debug else "info",
    )


def run_ui() -> None:
    from ui.dashboard import start_ui

    start_ui()


def run_all(debug: bool) -> None:
    api_thread = threading.Thread(
        target=run_api, args=(debug,), daemon=True, name="api"
    )
    ui_thread = threading.Thread(target=run_ui, daemon=True, name="ui")
    api_thread.start()
    ui_thread.start()
    run_voice(debug)


@click.command()
@click.option(
    "--mode",
    default="voice",
    type=click.Choice(["voice", "text", "api", "ui", "all"]),
    help="Mode de démarrage",
)
@click.option("--debug", is_flag=True, help="Mode debug (logs verbeux)")
def main(mode: str, debug: bool) -> None:
    """YadjiOs — Assistant IA personnel"""
    setup_logging(debug)
    print_banner()
    print_startup_status()

    if mode == "voice":
        run_voice(debug)
    elif mode == "text":
        run_text(debug)
    elif mode == "api":
        run_api(debug)
    elif mode == "ui":
        run_ui()
    elif mode == "all":
        run_all(debug)


if __name__ == "__main__":
    main()
