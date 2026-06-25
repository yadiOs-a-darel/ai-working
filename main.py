"""Point d'entrée principal YadjiOs."""

import os
import subprocess
import sys
import threading
import time
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
PROJECT_DIR = Path(__file__).resolve().parent


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


def print_startup_status(mode: str = "voice") -> None:
    from core.llm import get_llm_client

    light_mode = mode in ("text", "ui")

    if not light_mode:
        from core.stt import get_stt_engine
        from core.tts import get_tts_engine
        from core.wake_word import get_wake_word_detector
        from memory.long_term import get_long_term_memory
        from tools.registry import get_all_tools

        try:
            get_stt_engine()
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

    if light_mode:
        console.print("[dim]Mode léger — STT/TTS/ChromaDB non chargés[/dim]")
    elif info["ollama_available"] and not llm.model_supports_tools():
        console.print(
            "[yellow][!][/yellow] Outils désactivés : ce modèle ne supporte pas le tool calling "
            "(utilisez mistral ou un modèle compatible pour les outils)"
        )
    elif not settings.agent_tools_enabled:
        console.print(
            "[yellow][!][/yellow] Agent outils désactivé — chat direct Ollama "
            "(AGENT_TOOLS_ENABLED=true pour météo, web, etc.)"
        )

    if not light_mode:
        try:
            from memory.long_term import get_long_term_memory

            ltm = get_long_term_memory()
            console.print(
                f"[green][✓][/green] Mémoire vectorielle : ChromaDB "
                f"({ltm.count()} souvenirs)"
            )
        except Exception as exc:
            console.print(f"[yellow][!][/yellow] Mémoire long terme : {exc}")

        from tools.registry import get_all_tools

        tools = get_all_tools()
        console.print(f"[green][✓][/green] Tools chargés : {len(tools)} outils disponibles")

        from core.tts import get_tts_engine

        tts = get_tts_engine()
        speak_label = "activée" if settings.speak_responses else "désactivée"
        console.print(
            f"[green][✓][/green] TTS prêt : {settings.tts_engine} "
            f"({settings.tts_voice}, {settings.tts_voice_gender}) — lecture {speak_label}"
        )

        from core.wake_word import get_wake_word_detector

        wake = get_wake_word_detector()
        if wake.is_active:
            console.print(
                f'[green][✓][/green] Wake word actif : "{settings.wake_word}"'
            )
        else:
            console.print(
                f'[green][✓][/green] Écoute vocale : dites « {settings.wake_word} »'
            )

    if info["ollama_available"]:
        with console.status("[dim]Chargement du modèle en RAM...[/dim]", spinner="dots"):
            llm.prewarm_model()

    console.print(
        f"[green][✓][/green] Interface web : "
        f"http://localhost:{settings.ui_port}"
        + (" (processus séparé)" if mode == "text" and settings.ui_auto_start else "")
    )
    if mode == "text" and not settings.ui_auto_start:
        console.print(
            "[dim]Interface web désactivée au démarrage "
            "(UI_AUTO_START=true ou make start-ui)[/dim]"
        )
    if mode == "ui":
        return
    console.print("\n[bold]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold]")
    console.print(
        f"  [bold green]{settings.jarvis_name} en ligne. En attente...[/bold green]"
    )
    console.print("[bold]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold]\n")


def play_startup_greeting(mode: str) -> None:
    """Message d'accueil au démarrage (console + voix en mode vocal)."""
    if not settings.startup_greeting_enabled:
        return
    greeting = settings.startup_greeting.format(
        user_name=settings.user_name,
        jarvis_name=settings.jarvis_name,
    ).strip()
    if not greeting:
        return
    console.print(f"[bold green][{settings.jarvis_name}]:[/bold green] {greeting}")
    if mode in ("voice", "all") and settings.speak_responses:
        try:
            from core.tts import get_tts_engine

            get_tts_engine(reload=True).speak(greeting)
        except Exception:
            logger.warning("Accueil vocal ignoré (TTS indisponible)")


def run_voice(debug: bool) -> None:
    from core.pipeline import get_pipeline

    play_startup_greeting("voice")
    pipeline = get_pipeline(debug=debug)
    pipeline.run()


def _start_ui_process() -> subprocess.Popen:
    """Lance Gradio dans un sous-processus (évite les bugs asyncio en thread)."""
    return subprocess.Popen(
        [sys.executable, str(PROJECT_DIR / "main.py"), "--mode", "ui"],
        cwd=str(PROJECT_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _stop_ui_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def run_text(debug: bool) -> None:
    import time as time_mod

    from core.pipeline import get_pipeline

    ui_proc = None
    if settings.ui_auto_start:
        ui_proc = _start_ui_process()
        time.sleep(3)

    pipeline = get_pipeline(debug=debug, text_only=True)
    if settings.ui_auto_start:
        console.print(
            f"[cyan]Mode texte + web : http://localhost:{settings.ui_port}[/cyan]\n"
        )
    else:
        console.print(
            "[cyan]Mode texte — tapez 'quit' pour quitter[/cyan]\n"
            "[dim]Interface web : make start-ui (dans un autre terminal)[/dim]\n"
        )
    try:
        while True:
            try:
                text = input("[Vous]: ").strip()
                if text.lower() in ("quit", "exit", "q"):
                    break
                if not text:
                    continue
                t0 = time_mod.perf_counter()
                console.print(f"[green][{settings.jarvis_name}]:[/green] ", end="")
                response = pipeline.process_text(
                    text, vocal_mode=False, speak=False, stream=True
                )
                elapsed = time_mod.perf_counter() - t0
                console.print(f"[dim]({elapsed:.1f}s)[/dim]\n")
            except (KeyboardInterrupt, EOFError):
                break
    finally:
        _stop_ui_process(ui_proc)


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
    play_startup_greeting("all")
    api_thread = threading.Thread(
        target=run_api, args=(debug,), daemon=True, name="api"
    )
    ui_proc = _start_ui_process()
    api_thread.start()
    time.sleep(2)
    try:
        run_voice(debug)
    finally:
        _stop_ui_process(ui_proc)


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
    print_startup_status(mode=mode)

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
