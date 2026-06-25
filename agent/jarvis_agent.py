"""Agent LangChain principal JARVIS."""

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Iterator, Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from loguru import logger

from agent.prompts import get_compact_system_prompt, get_system_prompt
from config.settings import settings
from core.llm import get_llm_client
from memory.long_term import get_long_term_memory
from memory.short_term import ShortTermMemory
from tools.registry import get_all_tools

_agent_instance: Optional["JarvisAgent"] = None


class JarvisAgent:
    def __init__(self, session_id: str = "default") -> None:
        self.session_id = session_id
        self.llm_client = get_llm_client()
        self.memory = ShortTermMemory(session_id)
        self.long_term = get_long_term_memory()
        self.tools = get_all_tools()
        self._executor: Optional[AgentExecutor] = None
        self._build_agent()

    def _build_agent(self, system_prompt: str | None = None) -> None:
        system_prompt = system_prompt or get_system_prompt()
        if not self.llm_client.model_supports_tools():
            self._executor = None
            return

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        try:
            agent = create_tool_calling_agent(
                self.llm_client.chat_model,
                self.tools,
                prompt,
            )
            self._executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=False,
                handle_parsing_errors=True,
                max_iterations=5,
            )
        except Exception:
            logger.exception("Erreur construction agent, mode LLM simple")
            self._executor = None

    def _get_chat_history(self) -> list:
        messages = []
        for msg in self.memory.get_messages()[-10:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                from langchain_core.messages import AIMessage

                messages.append(AIMessage(content=msg["content"]))
        return messages

    def _invoke_agent(self, user_input: str) -> dict:
        return self._executor.invoke(
            {
                "input": user_input,
                "chat_history": self._get_chat_history(),
            }
        )

    def chat(self, user_input: str, vocal_mode: bool = False) -> str:
        """Retourne la réponse texte de l'agent."""
        start = time.perf_counter()
        context = self.long_term.get_relevant_context(user_input)
        system_with_context = get_system_prompt(vocal_mode, context)
        self._build_agent(system_with_context)

        use_agent = (
            settings.agent_tools_enabled
            and self._executor is not None
        )

        try:
            if use_agent:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(self._invoke_agent, user_input)
                    try:
                        result = future.result(timeout=settings.ollama_timeout)
                    except FuturesTimeout as exc:
                        raise TimeoutError(
                            f"Agent sans réponse en {settings.ollama_timeout}s — "
                            "passez en mode chat simple (AGENT_TOOLS_ENABLED=false)."
                        ) from exc
                response = result.get("output", "")
                tools_used = [
                    step[0].tool
                    for step in result.get("intermediate_steps", [])
                    if step
                ]
                if tools_used:
                    logger.info(f"Outils utilisés : {tools_used}")
            else:
                compact_prompt = get_compact_system_prompt(context)
                response = self._simple_chat(user_input, compact_prompt)

            duration = (time.perf_counter() - start) * 1000
            logger.info(f"Réponse générée en {duration:.0f}ms")

            self.memory.add_user_message(user_input)
            self.memory.add_ai_message(response)
            self.long_term.save_exchange(user_input, response)

            return response
        except TimeoutError as exc:
            logger.warning(str(exc))
            return str(exc)
        except Exception:
            logger.exception("Erreur agent, fallback LLM simple")
            try:
                response = self._simple_chat(
                    user_input, get_compact_system_prompt(context)
                )
                self.memory.add_user_message(user_input)
                self.memory.add_ai_message(response)
                return response
            except Exception as fallback_exc:
                if isinstance(fallback_exc, TimeoutError):
                    return str(fallback_exc)
                return f"Désolé, une erreur s'est produite : {fallback_exc}"

    def _simple_chat(self, user_input: str, system_prompt: str) -> str:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]
        for msg in self.memory.get_messages()[-10:]:
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["content"]})
        messages.append({"role": "user", "content": user_input})
        return self.llm_client.chat_local(messages)

    def chat_stream(self, user_input: str) -> Iterator[str]:
        """Stream token par token."""
        context = self.long_term.get_relevant_context(user_input)
        system_prompt = get_system_prompt(extra_context=context)
        model = self.llm_client.chat_model

        messages = [
            SystemMessage(content=system_prompt),
            *self._get_chat_history(),
            HumanMessage(content=user_input),
        ]

        full_response = ""
        try:
            for chunk in model.stream(messages):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    full_response += token
                    yield token
        except Exception:
            logger.exception("Erreur streaming")
            fallback = self.chat(user_input)
            yield fallback
            return

        self.memory.add_user_message(user_input)
        self.memory.add_ai_message(full_response)
        self.long_term.save_exchange(user_input, full_response)


def get_jarvis_agent(session_id: str = "default") -> JarvisAgent:
    global _agent_instance
    if _agent_instance is None or _agent_instance.session_id != session_id:
        _agent_instance = JarvisAgent(session_id)
    return _agent_instance
