"""在终端体验本地 Qwen 的多轮会话和增量预填充。"""

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
            raise RuntimeError("本地服务还没有加载 /chat 接口，请重启项目的 HTTP 服务。") from error
        try:
            detail = json.load(error).get("error", error.reason)
        except (AttributeError, ValueError):
            detail = error.reason
        raise RuntimeError(detail) from error
    except OSError as error:
        raise RuntimeError(f"无法访问本地 {port} 端口：{error.reason if hasattr(error, 'reason') else error}") from error


def post(port, body):
    with chat_response(port, body) as response:
        return json.load(response)


def stream_chat(port, body, on_delta):
    with chat_response(port, {**body, "stream": True}) as response:
        if response.headers.get_content_type() != "application/x-ndjson":
            raise RuntimeError("本地服务还没有加载流式聊天接口。")
        for line in response:
            event = json.loads(line)
            if "error" in event:
                raise RuntimeError(event["error"])
            if event.get("done"):
                return event
            on_delta(event["delta"])
    raise RuntimeError("聊天流在本轮完成前中断。")


def run(port, limit, prompts):
    session_id = uuid4().hex
    messages = []

    def reset():
        post(port, {"session_id": session_id, "reset": True})
        messages.clear()

    if prompts is None:
        print("流式聊天已启动。每轮显示服务实际使用的解码调度。/reset 新会话，/quit 退出。", flush=True)
    pending = iter(prompts) if prompts is not None else None
    try:
        while True:
            if pending is None:
                prompt = input("\n你 > ").strip()
            else:
                prompt = next(pending, None)
                if prompt is None:
                    break
                print(f"\n你 > {prompt}", flush=True)
            if not prompt:
                continue
            if prompt == "/quit":
                break
            if prompt == "/reset":
                try:
                    reset()
                    print("已清空会话。", flush=True)
                except RuntimeError as error:
                    print(error, file=sys.stderr)
                continue
            turn = {"role": "user", "content": prompt}
            try:
                print("模型 > ", end="", flush=True)
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
            print(f"  [第 {len(messages) // 2} 轮 · 复用 {stats['cached_prompt_tokens']}/{stats['input_tokens']} token"
                  f" · 新算 {stats['prefill_tokens']} token · 首 token {stats['ttft_s']:.2f}s"
                  f" · 解码 {stats['decode_tok_s']:.1f} token/s"
                  f" · {stats['draft_schedule']} · 验证 {stats['target_decode_passes']} 轮]", flush=True)
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
    parser.add_argument("--port", type=int, default=8765, help="本地 HTTP 服务端口")
    parser.add_argument("--max-tokens", type=int, default=128, help="每轮最多生成的 token 数")
    parser.add_argument("--prompt", action="append", help="按顺序执行一轮，可重复传入")
    args = parser.parse_args()
    if args.max_tokens < 1 or not 1 <= args.port <= 65535:
        parser.error("--max-tokens 必须为正整数，--port 必须在 1–65535 之间")
    raise SystemExit(run(args.port, args.max_tokens, args.prompt))


if __name__ == "__main__":
    main()
