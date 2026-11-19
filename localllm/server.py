"""Local HTTP endpoint backed by a single resident Qwen text model."""
import argparse
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
from time import perf_counter

import mlx.core as mx
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream

from .chat_session import ChatSession
from .cli import DRAFT_SCHEDULES, OUTPUT_TOKENS, chat_tokens, generate, nonnegative, positive, prompt_tokens
from .draft_short_head import prepare_short_head
from .memory import DEFAULT_CACHE_GIB, DEFAULT_MEMORY_GIB, GIB, MemoryBudgetError, metal_job
from .model import COMPILED_DELTA_WIDTHS, COMPILED_TREE_WIDTHS, LayerState, Model
from .profiles import OPTIMIZED_DRAFT, OPTIMIZED_MODEL
from .spec_tree import VerificationTree


WEB_DIR = Path(__file__).with_name("web")
WEB_ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/chat-ui.css": ("chat.css", "text/css; charset=utf-8"),
    "/chat-ui.js": ("chat.js", "text/javascript; charset=utf-8"),
    "/markdown-it.min.js": ("markdown-it.min.js", "text/javascript; charset=utf-8"),
}


class ChatStream:
    def __init__(self, handler, tokenizer):
        self.handler = handler
        self.tokenizer = tokenizer
        self.decoder = DecodeStream(skip_special_tokens=True)
        self.started = False
        self.sent = []

    def send(self, event):
        if not self.started:
            self.handler.close_connection = True
            self.handler.send_response(200)
            self.handler.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.handler.send_header("Cache-Control", "no-store")
            self.handler.end_headers()
            self.started = True
        self.handler.wfile.write((json.dumps(event, ensure_ascii=False) + "\n").encode())
        self.handler.wfile.flush()

    def token(self, token):
        text = self.decoder.step(self.tokenizer, token)
        if text:
            self.send({"delta": text})
            self.sent.append(text)

    def finish(self, text, stats):
        # The full decoder may emit a replacement character for a trailing
        # partial UTF-8 sequence when the token budget is exhausted.
        prefix = "".join(self.sent)
        if text.startswith(prefix) and len(text) > len(prefix):
            self.send({"delta": text[len(prefix):]})
        self.send({"done": True, "text": text, "metrics": stats})


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, body):
        encoded = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        asset = WEB_ASSETS.get(self.path)
        if asset is not None:
            name, content_type = asset
            try:
                content = (WEB_DIR / name).read_bytes()
            except FileNotFoundError:
                self.reply(404, {"error": "Chat page asset is missing"})
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if self.path == "/":
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                    "img-src data:; connect-src 'self'; frame-src 'self' about:; "
                    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
                )
            self.end_headers()
            with suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(content)
        elif self.path == "/healthz":
            self.reply(200, {
                "ready": True, "model": "Qwen/Qwen3.8-27B", "dtype": "BF16",
                "profile": self.server.options.profile,
                "default_draft_schedule": self.server.options.draft_schedule,
            })
        else:
            self.reply(404, {"error": "Unknown endpoint"})

    def do_POST(self):
        if self.path == "/chat":
            self.do_chat()
            return
        if self.path != "/generate":
            self.reply(404, {"error": "Unknown endpoint"})
            return
        args = self.server.options
        try:
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            prompt = data["prompt"]
            limit = data.get("max_tokens", OUTPUT_TOKENS)
            thinking = data.get("thinking", False)
            schedule = data.get("draft_schedule", args.draft_schedule)
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("prompt must be a nonempty string")
            if type(limit) is not int or limit < 1:
                raise ValueError("max_tokens must be a positive integer")
            if type(thinking) is not bool:
                raise ValueError("thinking must be a boolean")
            if schedule not in DRAFT_SCHEDULES:
                raise ValueError(f"draft_schedule must be one of {', '.join(DRAFT_SCHEDULES)}")
            model = self.server.model
            tokenizer = self.server.tokenizer
            ids = prompt_tokens(tokenizer, args.model, prompt, thinking)
            if len(ids) + limit > model.config["max_position_embeddings"]:
                raise ValueError("prompt and max_tokens exceed the model context")
        except (KeyError, TypeError, ValueError) as error:
            self.reply(400, {"error": str(error)})
            return
        try:
            self.server.chat_session = None
            self.server.chat_id = None
            result, stats = generate(
                model, ids, limit, args.chunk, stop=self.server.eos, budget=self.server.budget,
                draft_schedule=schedule, **self.server.draft_options,
            )
        except MemoryBudgetError as error:
            self.reply(503, {"error": str(error)})
            return
        finally:
            mx.clear_cache()
        self.reply(200, {"text": tokenizer.decode(result, skip_special_tokens=True), "metrics": stats})

    def do_chat(self):
        try:
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not isinstance(data, dict):
                raise ValueError("expected a JSON object")
            key = data["session_id"]
            if not isinstance(key, str) or not 1 <= len(key) <= 64:
                raise ValueError("session_id must contain between 1 and 64 characters")
            if data.get("reset") is True:
                if self.server.chat_id == key:
                    self.server.chat_session = None
                    self.server.chat_id = None
                    mx.clear_cache()
                self.reply(200, {"reset": True})
                return
            messages = data["messages"]
            limit = data.get("max_tokens", 128)
            streaming = data.get("stream", False)
            if type(limit) is not int or limit < 1:
                raise ValueError("max_tokens must be a positive integer")
            if type(streaming) is not bool:
                raise ValueError("stream must be a boolean")
            if not isinstance(messages, list) or len(messages) % 2 != 1:
                raise ValueError("messages must contain alternating user and assistant turns, ending with user")
            for i, message in enumerate(messages):
                role = "assistant" if i % 2 else "user"
                if (not isinstance(message, dict) or message.get("role") != role
                        or not isinstance(message.get("content"), str)
                        or (role == "user" and not message["content"].strip())):
                    raise ValueError(f"invalid {role} message at index {i}")
            messages = [{"role": message["role"], "content": message["content"],
                         "reasoning_content": "", "tool_calls": []} for message in messages]
            args = self.server.options
            model, tokenizer = self.server.model, self.server.tokenizer
            ids = chat_tokens(tokenizer, args.model, messages, False)
            if len(ids) + limit > model.config["max_position_embeddings"]:
                raise ValueError("chat history and max_tokens exceed the model context")
        except (KeyError, TypeError, ValueError) as error:
            self.reply(400, {"error": str(error)})
            return
        if self.server.chat_id != key or self.server.chat_session is None:
            self.server.chat_session = ChatSession()
            self.server.chat_id = key
        stream = ChatStream(self, tokenizer) if streaming else None
        with suppress(BrokenPipeError, ConnectionResetError):
            try:
                result, stats = generate(
                    model, ids, limit, args.chunk, stop=self.server.eos, budget=self.server.budget,
                    draft_schedule=args.draft_schedule, session=self.server.chat_session,
                    on_token=stream.token if stream is not None else None,
                    **self.server.draft_options,
                )
                text = tokenizer.decode(result, skip_special_tokens=True)
                if stream is None:
                    self.reply(200, {"text": text, "metrics": stats})
                else:
                    stream.finish(text, stats)
            except MemoryBudgetError as error:
                if stream is not None and stream.started:
                    stream.send({"error": str(error)})
                else:
                    self.reply(503, {"error": str(error)})
            finally:
                mx.clear_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("weights/Qwen3.8-27B"))
    parser.add_argument("--profile", choices=("reference", "optimized"), default="reference",
                        help="select the reference path or the validated BF16 performance preset")
    parser.add_argument("--draft-schedule", choices=DRAFT_SCHEDULES, default="standard",
                        help="schedule for requests that omit draft_schedule")
    parser.add_argument("--port", type=positive, default=8765)
    parser.add_argument("--chunk", type=positive, default=512)
    parser.add_argument("--group-rows", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--skinny", action="store_true")
    parser.add_argument("--reserve-kv", action="store_true")
    parser.add_argument("--vector-bf16", action="store_true")
    parser.add_argument("--memory-gib", type=positive, default=DEFAULT_MEMORY_GIB)
    parser.add_argument("--cache-gib", type=nonnegative, default=DEFAULT_CACHE_GIB)
    parser.add_argument("--draft-depth", type=int, choices=range(9), default=0)
    parser.add_argument("--lookup-depth", type=int, choices=range(33), default=0)
    parser.add_argument("--draft-vocab", type=positive, default=0)
    args = parser.parse_args()
    if args.profile == "optimized" and (args.lookup_depth or args.draft_depth or args.draft_vocab):
        parser.error("the optimized profile supplies its own drafting settings")
    try:
        with metal_job(args.memory_gib, args.cache_gib) as budget:
            serve(args, budget)
    except MemoryBudgetError as error:
        parser.exit(1, f"{error}\n")


def warm_verification_graphs(model, budget):
    layers = [layer for layer in model.layers if layer.gdn and layer.compiled_delta is not None]
    reserve = GIB
    if not layers or mx.get_active_memory() + budget.cache_limit + reserve > budget.limit:
        return
    budget.check(force=True, required_bytes=reserve)
    start = perf_counter()
    config = model.config
    hk, hv = config["linear_num_key_heads"], config["linear_num_value_heads"]
    dk, dv = config["linear_key_head_dim"], config["linear_value_head_dim"]
    taps = config["linear_conv_kernel_dim"] - 1
    conv = mx.contiguous(mx.zeros((taps, 2 * hk * dk + hv * dv), dtype=mx.bfloat16))
    delta = mx.contiguous(mx.zeros((hv, dv, dk), dtype=mx.float32))
    mx.eval(conv, delta)
    for width in sorted(set(COMPILED_DELTA_WIDTHS) | set(COMPILED_TREE_WIDTHS)):
        if width > model.short_tokens:
            continue
        hidden = mx.contiguous(mx.zeros((width, config["hidden_size"]), dtype=mx.bfloat16))
        trees = [None] if width in COMPILED_DELTA_WIDTHS else []
        if model.compact_tree and width in COMPILED_TREE_WIDTHS:
            for forks in (1, 2):
                spine = width - forks
                parents = [-1, *range(spine - 1), *range(forks)]
                trees.append(VerificationTree(parents, 0, taps, spine=spine))
        for tree in trees:
            for layer in layers:
                state = LayerState(conv=conv, delta=delta, compile_delta_tree=True)
                output = layer.forward(hidden, state, 0, keep_replay=True, tree=tree)
                replay = state.replay[2]
                mx.eval(output, state.conv, state.delta, replay.updates, replay.keys, replay.decay)
            budget.check()
    print(f"Verification graphs warmed in {perf_counter() - start:.3f} s", flush=True)


def serve(args, budget):
    with HTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.options = args
        server.budget = budget
        server.chat_id = None
        server.chat_session = None
        server.tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
        eos = json.loads((args.model / "generation_config.json").read_text())["eos_token_id"]
        server.eos = (eos,) if isinstance(eos, int) else tuple(eos)
        model_options = {
            "merge": args.merge, "skinny": args.skinny, "reserve_kv": args.reserve_kv,
            "vector_bf16": args.vector_bf16,
        }
        draft_options = {
            "draft_depth": args.draft_depth, "draft_vocab": args.draft_vocab, "lookup_depth": args.lookup_depth,
        }
        if args.profile == "optimized":
            model_options.update(OPTIMIZED_MODEL)
            draft_options.update(OPTIMIZED_DRAFT)
        server.draft_options = draft_options
        server.model = Model(args.model, group_rows=args.group_rows, mtp=bool(draft_options["draft_depth"]),
                             budget=budget, **model_options)
        if args.profile == "optimized":
            try:
                warm_verification_graphs(server.model, budget)
            except MemoryBudgetError:
                pass
            finally:
                mx.clear_cache()
            try:
                server.model.short_draft_head = prepare_short_head(server.model, budget)
            except MemoryBudgetError:
                mx.clear_cache()
        print(f"Qwen3.8-27B BF16 ready at http://127.0.0.1:{args.port} ({args.profile})", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
