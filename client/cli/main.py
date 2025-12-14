
"""MolSys-AI CLI.

This CLI is an API client for the MolSys-AI HTTP services.
"""

import argparse
import getpass
import sys
from typing import Dict, List, Optional

from cli.config import CLIConfig, load_config, save_config
from cli.http_api import (
    pick_base_urls,
    post_engine_chat,
    post_chat,
    post_chat_json,
    post_chat_json_any,
    post_chat_messages,
    resolve_api_key,
)


def _add_common_server_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--server-url",
        help="Override server base URL (default is taken from the local config).",
    )
    parser.add_argument(
        "--api-key",
        help="Override API key (default is taken from env or local config).",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="molsys-ai", description="MolSys-AI CLI")
    parser.add_argument("--version", action="store_true", help="Print version and exit")

    sub = parser.add_subparsers(dest="cmd")

    p_login = sub.add_parser("login", help="Store an API key locally")
    p_login.add_argument("--api-key", help="Provide the key directly (otherwise prompt).")
    p_login.add_argument(
        "--stdin",
        action="store_true",
        help="Read the API key from stdin (useful for non-interactive setups).",
    )

    p_logout = sub.add_parser("logout", help="Remove the stored API key")

    p_config = sub.add_parser("config", help="Show current CLI configuration")
    p_config.add_argument(
        "--all",
        action="store_true",
        help="Show advanced/internal fields.",
    )

    p_chat = sub.add_parser("chat", help="Chat with MolSys-AI (RAG-enabled, via /v1/chat)")
    _add_common_server_flags(p_chat)
    p_chat.add_argument("-m", "--message", help="Send a single message and exit.")
    p_chat.add_argument(
        "--system",
        help="Optional system prompt (inserted as the first message).",
    )

    p_agent = sub.add_parser(
        "agent",
        help="Local tool-using agent (executes tools locally, uses the remote LLM for generation)",
    )
    _add_common_server_flags(p_agent)
    p_agent.add_argument("-m", "--message", help="Send a single message and exit.")
    p_agent.add_argument(
        "--system",
        help="Optional system prompt (inserted as the first message).",
    )
    p_agent.add_argument(
        "--yes",
        action="store_true",
        help="Auto-approve tool execution (unsafe; intended for trusted environments).",
    )

    p_docs = sub.add_parser("docs", help="Ask a documentation question (forces sources, via /v1/chat)")
    _add_common_server_flags(p_docs)
    p_docs.add_argument("-m", "--message", help="Send a single question and exit (otherwise interactive).")
    p_docs.add_argument("-k", type=int, default=5, help="Number of retrieved snippets (default: 5).")

    p_tools = sub.add_parser("tools", help="Inspect local agent tools")
    p_tools_sub = p_tools.add_subparsers(dest="tools_cmd")
    p_tools_sub.add_parser("list", help="List available local tools")
    p_tools_sub.add_parser("doctor", help="Check optional tool dependencies")

    return parser


def _cmd_login(cfg: CLIConfig, api_key_arg: Optional[str], *, stdin: bool) -> int:
    if stdin:
        key = sys.stdin.read().strip()
    else:
        key = (api_key_arg or "").strip() or getpass.getpass("MolSys-AI API key: ").strip()
    if not key:
        print("No API key provided.", file=sys.stderr)
        return 1
    save_config(CLIConfig(**{**cfg.__dict__, "api_key": key}))
    print("API key stored.")
    return 0


def _cmd_logout(cfg: CLIConfig) -> int:
    if not cfg.api_key:
        print("No API key is currently stored.")
        return 0
    save_config(CLIConfig(**{**cfg.__dict__, "api_key": None}))
    print("API key removed.")
    return 0


def _cmd_config(cfg: CLIConfig, *, show_all: bool) -> int:
    masked = None
    if cfg.api_key:
        masked = cfg.api_key[:4] + "…" + cfg.api_key[-4:] if len(cfg.api_key) >= 12 else "****"
    print("MolSys-AI CLI config:")
    print(f"- public_base_url: {cfg.public_base_url}")
    print(f"- api_key: {masked}")
    if show_all:
        print(f"- lan_base_url: {cfg.lan_base_url}")
        print(f"- connect_timeout_s: {cfg.connect_timeout_s}")
        print(f"- read_timeout_s: {cfg.read_timeout_s}")
    return 0


def _cmd_chat(cfg: CLIConfig, *, server_url: str | None, api_key_arg: str | None, message: str | None, system: str | None) -> int:
    api_key = resolve_api_key(api_key_arg, cfg.api_key)
    if not api_key:
        print("Missing API key. Run `molsys-ai login` or set MOLSYS_AI_API_KEY.", file=sys.stderr)
        return 1

    public_base_url = (server_url or cfg.public_base_url).strip().rstrip("/")
    base_urls = pick_base_urls(api_key=api_key, public_base_url=public_base_url, lan_base_url=cfg.lan_base_url)

    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})

    def send_and_print(user_text: str) -> bool:
        nonlocal messages
        messages.append({"role": "user", "content": user_text})
        try:
            data, sources = post_chat_json_any(
                base_urls=base_urls,
                api_key=api_key,
                query=None,
                messages=messages,
                k=5,
                client="cli",
                rag="auto",
                sources="auto",
                timeout=cfg.timeout,
            )
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return False
        reply = data.get("answer")
        if not isinstance(reply, str) or not reply.strip():
            print("Error: server returned no 'answer'.", file=sys.stderr)
            return False
        messages.append({"role": "assistant", "content": reply})
        print(reply)
        if sources:
            print("\nSources:")
            for s in sources:
                sid = s.get("id")
                url = s.get("url")
                path = s.get("path")
                label = s.get("label")
                prefix = f"[{sid}] " if isinstance(sid, int) else "- "
                if isinstance(url, str) and url.strip():
                    print(f"{prefix}{url.strip()}")
                elif isinstance(path, str) and path.strip():
                    suffix = f"#{label}" if isinstance(label, str) and label.strip() else ""
                    print(f"{prefix}{path.strip()}{suffix}")
        return True

    if message:
        return 0 if send_and_print(message) else 1

    # Interactive mode.
    print("MolSys-AI chat (Ctrl-D or /exit to quit).")
    while True:
        try:
            user_text = input("> ").strip()
        except EOFError:
            print()
            break
        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            break
        if not send_and_print(user_text):
            return 1
    return 0


def _cmd_agent(
    cfg: CLIConfig,
    *,
    server_url: str | None,
    api_key_arg: str | None,
    message: str | None,
    system: str | None,
    assume_yes: bool,
) -> int:
    from agent.executor import create_default_executor
    from agent.planner import Plan, SimplePlanner

    api_key = resolve_api_key(api_key_arg, cfg.api_key)
    if not api_key:
        print("Missing API key. Run `molsys-ai login` or set MOLSYS_AI_API_KEY.", file=sys.stderr)
        return 1

    public_base_url = (server_url or cfg.public_base_url).strip().rstrip("/")
    base_urls = pick_base_urls(api_key=api_key, public_base_url=public_base_url, lan_base_url=cfg.lan_base_url)

    planner = SimplePlanner()
    executor = create_default_executor()

    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})

    def maybe_run_tool(plan: Plan) -> str | None:
        if not plan.use_tools or not plan.tool_name:
            return None

        tool_name = plan.tool_name
        tool_args = plan.tool_args or {}

        if not executor.has_tool(tool_name):
            if tool_name.startswith("molsysmt."):
                return (
                    f"Tool '{tool_name}' is not available in this environment.\n"
                    "It requires the relevant MolSysSuite toolchain. Install it locally and try again. "
                    "You can check with: `molsys-ai tools doctor`."
                )
            return f"Tool '{tool_name}' is not available in this environment. See `molsys-ai tools list`."

        if assume_yes:
            approved = True
        else:
            print(f"[agent] Proposed tool call: {tool_name}({tool_args})", file=sys.stderr)
            try:
                ans = input("Run this tool locally? [y/N] ").strip().lower()
            except EOFError:
                ans = ""
            approved = ans in {"y", "yes"}

        if not approved:
            return "Tool execution was declined by the user."

        try:
            out = executor.execute(tool_name, **tool_args)
        except Exception as exc:
            return f"Tool execution failed: {exc}"
        return str(out)

    def send_and_print(user_text: str) -> bool:
        nonlocal messages
        messages.append({"role": "user", "content": user_text})

        plan = planner.decide(messages, force_rag=False)
        tool_result = maybe_run_tool(plan)
        if tool_result is not None:
            # Use a system message for tool results to keep `/v1/chat` compatible
            # with chat templates that expect system/user/assistant roles only.
            messages.append({"role": "system", "content": f"Tool result:\n{tool_result}"})

        try:
            reply = post_engine_chat(base_urls=base_urls, api_key=api_key, messages=messages, timeout=cfg.timeout)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return False
        messages.append({"role": "assistant", "content": reply})
        print(reply)
        return True

    if message:
        return 0 if send_and_print(message) else 1

    print("MolSys-AI agent (Ctrl-D or /exit to quit).")
    while True:
        try:
            user_text = input("> ").strip()
        except EOFError:
            print()
            break
        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            break
        if not send_and_print(user_text):
            return 1
    return 0


def _cmd_docs(cfg: CLIConfig, *, server_url: str | None, api_key_arg: str | None, message: str | None, k: int) -> int:
    api_key = resolve_api_key(api_key_arg, cfg.api_key)
    public_base_url = (server_url or cfg.public_base_url).strip().rstrip("/")
    base_urls = pick_base_urls(api_key=api_key, public_base_url=public_base_url, lan_base_url=cfg.lan_base_url)

    messages: List[Dict[str, str]] = []

    def send_and_print(user_text: str) -> bool:
        nonlocal messages
        messages.append({"role": "user", "content": user_text})
        try:
            data, sources = post_chat_json_any(
                base_urls=base_urls,
                api_key=api_key,
                query=None,
                messages=messages,
                k=k,
                client="cli",
                rag="on",
                sources="on",
                timeout=cfg.timeout,
            )
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return False
        answer = data.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            print("Error: server returned no 'answer'.", file=sys.stderr)
            return False
        messages.append({"role": "assistant", "content": answer})
        print(answer)
        if sources:
            print("\nSources:")
            for s in sources:
                sid = s.get("id")
                url = s.get("url")
                path = s.get("path")
                prefix = f"[{sid}] " if isinstance(sid, int) else "- "
                if isinstance(url, str) and url.strip():
                    print(f"{prefix}{url.strip()}")
                elif isinstance(path, str) and path.strip():
                    print(f"{prefix}{path.strip()}")
        return True

    if message:
        try:
            data, sources = post_chat_json_any(
                base_urls=base_urls,
                api_key=api_key,
                query=message,
                messages=None,
                k=k,
                client="cli",
                rag="on",
                sources="on",
                timeout=cfg.timeout,
            )
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        answer = data.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            print("Error: server returned no 'answer'.", file=sys.stderr)
            return 1
        print(answer)
        if sources:
            print("\nSources:")
            for s in sources:
                sid = s.get("id")
                url = s.get("url")
                path = s.get("path")
                prefix = f"[{sid}] " if isinstance(sid, int) else "- "
                if isinstance(url, str) and url.strip():
                    print(f"{prefix}{url.strip()}")
                elif isinstance(path, str) and path.strip():
                    print(f"{prefix}{path.strip()}")
        return 0

    print("MolSys-AI docs chat (Ctrl-D or /exit to quit).")
    while True:
        try:
            user_text = input("> ").strip()
        except EOFError:
            print()
            break
        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            break
        if not send_and_print(user_text):
            return 1
    return 0


def _cmd_tools_list() -> int:
    from agent.executor import create_default_executor

    executor = create_default_executor()
    tools = sorted(executor.tools.values(), key=lambda t: t.name)

    if not tools:
        print("No local tools are available in this environment.")
        return 0

    print("Available local tools:")
    for tool in tools:
        desc = tool.description.strip() if tool.description else ""
        if desc:
            print(f"- {tool.name}: {desc}")
        else:
            print(f"- {tool.name}")
    return 0


def _cmd_tools_doctor() -> int:
    import importlib.util as iu

    def status(name: str) -> str:
        return "OK" if iu.find_spec(name) is not None else "MISSING"

    print("Local tool dependencies:")
    print(f"- molsysmt: {status('molsysmt')}")

    if iu.find_spec("molsysmt") is None:
        print()
        print("Install hint (recommended in a dedicated conda env for the local agent):")
        print("- conda create -n molsys-agent python=3.12 -c conda-forge")
        print("- conda activate molsys-agent")
        print("- conda install -c uibcdf -c conda-forge molsysmt")
        print("- pip install molsys-ai  # installs the CLI/agent (lightweight)")
        print()
        print("Note: avoid installing MolSysSuite toolchains into the vLLM inference env used on the server.")
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print("MolSys-AI CLI")
        return 0

    cfg = load_config()

    if args.cmd == "login":
        return _cmd_login(cfg, args.api_key, stdin=bool(getattr(args, "stdin", False)))
    if args.cmd == "logout":
        return _cmd_logout(cfg)
    if args.cmd == "config":
        return _cmd_config(cfg, show_all=bool(getattr(args, "all", False)))
    if args.cmd == "chat":
        return _cmd_chat(
            cfg,
            server_url=getattr(args, "server_url", None),
            api_key_arg=getattr(args, "api_key", None),
            message=getattr(args, "message", None),
            system=getattr(args, "system", None),
        )
    if args.cmd == "agent":
        return _cmd_agent(
            cfg,
            server_url=getattr(args, "server_url", None),
            api_key_arg=getattr(args, "api_key", None),
            message=getattr(args, "message", None),
            system=getattr(args, "system", None),
            assume_yes=bool(getattr(args, "yes", False)),
        )
    if args.cmd == "docs":
        return _cmd_docs(
            cfg,
            server_url=getattr(args, "server_url", None),
            api_key_arg=getattr(args, "api_key", None),
            message=getattr(args, "message", None),
            k=int(getattr(args, "k", 5)),
        )
    if args.cmd == "tools":
        if getattr(args, "tools_cmd", None) in {None, "list"}:
            return _cmd_tools_list()
        if getattr(args, "tools_cmd", None) == "doctor":
            return _cmd_tools_doctor()
        print("Unknown tools subcommand.", file=sys.stderr)
        return 2

    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
