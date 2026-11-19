"""Chat with a local Qwen model using streaming and incremental prefill."""

import argparse
from contextlib import contextmanager
import json
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4


@contextmanager
def chat_response(port, body):
    request = Request(
        f"http://127.0.0.1:{port}/chat",
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=300) as response:
            yield response
    except HTTPError as error:
        if error.code == 404:
            raise RuntimeError("The local server has no /chat endpoint. Restart the UltraRun server.") from error
        try:
            detail = json.load(error).get("error", error.reason)
        except (AttributeError, ValueError):
            detail = error.reason
        raise RuntimeError(detail) from error
    except OSError as error:
        detail = error.reason if hasattr(error, "reason") else error
        raise RuntimeError(f"Could not connect to the local server on port {port} ({detail})") from error


def post(port, body):
    with chat_response(port, body) as response:
        return json.load(response)


def stream_chat(port, body, on_delta):
    with chat_response(port, {**body, "stream": True}) as response:
        if response.headers.get_content_type() != "application/x-ndjson":
            raise RuntimeError("The local server did not return a chat stream.")
        for line in response:
            event = json.loads(line)
            if "error" in event:
                raise RuntimeError(event["error"])
            if event.get("done"):
                return event
            on_delta(event["delta"])
    raise RuntimeError("The chat stream ended before the turn was complete.")


def run(port, limit, prompts):
    session_id = uuid4().hex
    messages = []

    def reset():
        post(port, {"session_id": session_id, "reset": True})
        messages.clear()

    if prompts is None:
        print("Streaming chat is ready. Each turn shows the draft schedule used. /reset starts a new chat. /quit exits.",
              flush=True)
    pending = iter(prompts) if prompts is not None else None
    try:
        while True:
            if pending is None:
                prompt = input("\nYou > ").strip()
            else:
                prompt = next(pending, None)
                if prompt is None:
                    break
                print(f"\nYou > {prompt}", flush=True)
            if not prompt:
                continue
            if prompt == "/quit":
                break
            if prompt == "/reset":
                try:
                    reset()
                    print("Chat cleared.", flush=True)
                except RuntimeError as error:
                    print(error, file=sys.stderr)
                continue
            turn = {"role": "user", "content": prompt}
            try:
                print("Qwen > ", end="", flush=True)
                response = stream_chat(port, {
                    "session_id": session_id, "messages": [*messages, turn], "max_tokens": limit,
                }, lambda text: print(text, end="", flush=True))
            except RuntimeError as error:
                print(f"\n{error}", file=sys.stderr)
                if pending is not None or not messages:
                    return 1
                continue
            text, stats = response["text"], response["metrics"]
            messages.extend((turn, {"role": "assistant", "content": text}))
            print()
            print(f"  [Turn {len(messages) // 2} · Reused {stats['cached_prompt_tokens']}/{stats['input_tokens']} tokens"
                  f" · Prefilled {stats['prefill_tokens']} tokens · First token {stats['ttft_s']:.2f} s"
                  f" · Decode {stats['decode_tok_s']:.1f} tokens/s"
                  f" · {stats['draft_schedule']} · Verification rounds {stats['target_decode_passes']}]", flush=True)
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        if messages:
            try:
                reset()
            except RuntimeError:
                pass
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765, help="port of the local HTTP server")
    parser.add_argument("--max-tokens", type=int, default=128, help="maximum output tokens per turn")
    parser.add_argument("--prompt", action="append", help="send a prompt as one turn (repeat for multiple turns)")
    args = parser.parse_args()
    if args.max_tokens < 1 or not 1 <= args.port <= 65535:
        parser.error("--max-tokens must be a positive integer and --port must be between 1 and 65535")
    raise SystemExit(run(args.port, args.max_tokens, args.prompt))


if __name__ == "__main__":
    main()
