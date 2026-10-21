# server.py — cline2api as a local Python API (Flask).
#
# Python port of worker.js. 逆向自 https://github.com/luawei1/cline2api (Go 版反向代理)
#
# 核心逻辑：
#  1. 每次请求用 refreshToken 换 accessToken（缓存到内存，过期自动刷新）
#  2. 把 OpenAI / Anthropic 请求转发到 https://api.cline.bot/api/v1/chat/completions
#  3. SSE 流式响应剥掉上游 {data:{...}} 包装，透传给客户端
#  4. 账号池：多账号轮换 + 额度/限流冷却 + 串行队列（上游并发>1 返回空响应）
#
# 环境变量：
#  - CLINE_REFRESH_TOKEN (必需，多账号每行一个)
#  - API_KEY             (可选；未设置时使用内置默认 key "cline2api-default-key"，
#                         设为空字符串则完全关闭鉴权)
#
# 用法（OpenAI 兼容）：
#   curl http://localhost:8080/v1/chat/completions \
#     -H "Authorization: Bearer <API_KEY>" \
#     -H "Content-Type: application/json" \
#     -d '{"model":"cline-free/mimo-v2.6-flash","messages":[{"role":"user","content":"hi"}]}'
#
# Run:  python server.py
#       gunicorn -w 1 -b 0.0.0.0:8080 server:app   # keep 1 worker: in-memory account state
#
# Deps: flask + httpx.

import json
import os
import random
import re
import threading
import time
import uuid

import httpx
from flask import Flask, Response, request

CLINE_API_BASE = "https://api.cline.bot/api/v1"
VERSION = "1.1.8-py"

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization, x-api-key, anthropic-version, anthropic-beta",
}

# 模型列表：原样使用 Cline /v1/models 返回的完整模型 ID。
# 不人为添加 cline/ 前缀；Telegram 会完整显示这些 ID，避免不同供应商模型名被截断后混淆。
FALLBACK_MODELS = [
    {"id": "cline-free/nemotron-3.5-super-vl", "upstream": "cline-free/nemotron-3.5-super-vl", "provider": "nvidia", "cost": "free"},
    {"id": "z-ai/glm-5.3-flash", "upstream": "z-ai/glm-5.3-flash", "provider": "zai", "cost": "free"},
    {"id": "deepseek/deepseek-v4-flash", "upstream": "deepseek/deepseek-v4-flash", "provider": "deepseek", "cost": "free"},
    {"id": "poolside/laguna-s-2.1:free", "upstream": "poolside/laguna-s-2.1:free", "provider": "poolside", "cost": "free"},
    {"id": "cline-pass/glm-5.2", "upstream": "cline-pass/glm-5.2", "provider": "zai", "cost": "pass"},
    {"id": "cline-pass/deepseek-v4-flash", "upstream": "cline-pass/deepseek-v4-flash", "provider": "deepseek", "cost": "pass"},
    {"id": "cline-pass/qwen3.7-max", "upstream": "cline-pass/qwen3.7-max", "provider": "qwen", "cost": "pass"},
]

# Cline 官方免费模型白名单（2026-08-29，来自插件 recommended-models 接口）
FREE_WHITELIST = [
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v4-flash-0731",
    "z-ai/glm-5.3-flash",
    "z-ai/glm-5.2:free",
    "xiaomi/mimo-v2.5",
    "minimax/minimax-m3",
    "poolside/laguna-s-2.1",
    "cline-free/muse-spark-1.3-contributor",
    "cline-free/solar-pro4",
]

# 默认模型：cline-free/nemotron-3.5-super-vl（cline-free 官方免费额度，无需 credits）。
# 原 cline-free/deepseek-v4.1-flash 已被上游下线（404 model not found），勿再设为默认。
DEFAULT_MODEL = "cline-free/nemotron-3.5-super-vl"

MODELS_TTL = 10 * 60  # 10 分钟

_client = httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0))

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Token 管理（账号池）
# ---------------------------------------------------------------------------

_accounts = []
_account_index = 0
_current_account = None
_accounts_lock = threading.Lock()


def _log(msg):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        # Windows consoles (cp1252 etc.) can't render the Chinese log lines
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def parse_accounts():
    """从环境变量解析账号池：CLINE_REFRESH_TOKEN 每行一个。"""
    global _accounts
    raw = os.environ.get("CLINE_REFRESH_TOKEN") or ""
    tokens = [s.strip() for s in raw.split("\n") if len(s.strip()) > 8]
    if not tokens:
        return []
    with _accounts_lock:
        # Compare against the token each account was BUILT with (originalToken),
        # not the live refreshToken — Cline rotates refresh tokens on every
        # refresh, and comparing against the rotated value would rebuild the
        # pool on every request, wiping cooldowns and cached access tokens
        # (a subtle bug the original Worker has as well).
        changed = (len(_accounts) != len(tokens)
                   or any(_accounts[i].get("originalToken") != tokens[i] for i in range(len(tokens))))
        if changed:
            _accounts = [{"refreshToken": rt, "originalToken": rt, "accessToken": None,
                          "expiry": 0.0, "cooldownUntil": 0.0} for rt in tokens]
        # NOTE: return the shared dicts (not copies) so cooldown updates made
        # by callers mutate the pool in place, like the Worker's account objects.
        return _accounts


def get_account_token(account):
    """取得账号的 accessToken（独立缓存，失效/冷却则刷新）。"""
    global _current_account
    now = _now_ms()
    if account["cooldownUntil"] > now:
        raise RuntimeError("account_cooldown")
    if account["accessToken"] and now < account["expiry"]:
        return account["accessToken"]
    resp = _client.post(CLINE_API_BASE + "/auth/refresh", json={
        "refreshToken": account["refreshToken"],
        "grantType": "refresh_token",
    })
    if resp.status_code >= 400:
        # 刷新失败：冷却 60s，交给上层切号
        account["cooldownUntil"] = now + 60 * 1000
        raise RuntimeError("refresh_failed")
    data = resp.json()
    access_token = (data or {}).get("data", {}).get("accessToken")
    if not access_token:
        account["cooldownUntil"] = now + 60 * 1000
        raise RuntimeError("refresh_no_token")
    account["accessToken"] = access_token
    # Cline 会在刷新时轮换 refreshToken；必须保存新 token，避免下一次刷新 invalid_grant。
    new_rt = (data.get("data") or {}).get("refreshToken")
    if isinstance(new_rt, str) and new_rt.strip():
        account["refreshToken"] = new_rt.strip()
    # 过期时间：优先服务端，兜底 10 分钟，留 60s 余量
    expires_at = (data.get("data") or {}).get("expiresAt")
    expiry = now + 10 * 60 * 1000
    if isinstance(expires_at, (int, float)):
        expiry = expires_at
    elif isinstance(expires_at, str):
        try:
            from email.utils import parsedate_to_datetime
            import calendar
            expiry = calendar.timegm(parsedate_to_datetime(expires_at).utctimetuple()) * 1000
        except (ValueError, TypeError):
            try:
                expiry = int(float(expires_at))
            except (ValueError, TypeError):
                expiry = now + 10 * 60 * 1000
    account["expiry"] = expiry - 60000
    _current_account = account
    return access_token


def get_access_token():
    """轮询选择一个可用账号；最多尝试 pool.length 次（跳过冷却/刷新失败的）。"""
    pool = parse_accounts()
    if not pool:
        raise RuntimeError("缺少 CLINE_REFRESH_TOKEN 环境变量")
    for attempt in range(len(pool)):
        acc = pool[attempt % len(pool)]
        if acc["cooldownUntil"] and acc["cooldownUntil"] > _now_ms():
            continue
        try:
            return get_account_token(acc)
        except RuntimeError as exc:
            if str(exc) == "account_cooldown":
                continue
            continue  # 刷新失败也切下个号
    # 全部失败，清冷却重试一次最早的
    acc = pool[0]
    acc["cooldownUntil"] = 0
    try:
        return get_account_token(acc)
    except RuntimeError:
        raise RuntimeError("所有账号刷新 token 均失败")


def _now_ms():
    return time.time() * 1000


# Cline 客户端指纹请求头（官方靠这些头识别"是不是 Cline 客户端"）
# 缺少会被 403: "deepseek/deepseek-v4-flash is only available via Cline product surfaces"
def cline_headers(session_id, token):
    return {
        "Authorization": "Bearer workos:" + token,
        "Content-Type": "application/json",
        "User-Agent": "Cline/3.0.47",
        "HTTP-Referer": "https://cline.bot",
        "X-Title": "Cline",
        "X-IS-MULTIROOT": "false",
        "X-CLIENT-TYPE": "cline-sdk",
        "X-CLIENT-VERSION": "3.0.47",
        "X-PLATFORM": "terminal",
        "X-PLATFORM-VERSION": "3.0.47",
        "X-CORE-VERSION": "0.0.66",
        "X-Task-ID": session_id,
    }


def cline_fetch(path, body_obj, session_id, retried=False):
    token = get_access_token()
    headers = cline_headers(session_id, token)
    resp = _client.post(CLINE_API_BASE + path, headers=headers,
                        content=json.dumps(body_obj))
    if resp.status_code == 401 and not retried:
        # token 失效：标记当前账号冷却，强制重试（会用别的账号/刷新）
        global _current_account
        acc = _current_account
        if acc is not None:
            acc["cooldownUntil"] = _now_ms() + 60 * 1000
            acc["accessToken"] = None
            acc["expiry"] = 0
        return cline_fetch(path, body_obj, session_id, retried=True)
    return resp


# ---------------------------------------------------------------------------
# 并发限流队列：上游免费通道并发超过 1 就返回空响应，这里强制串行 + 间隔
# ---------------------------------------------------------------------------

_queue_lock = threading.Lock()
_queue_condition = threading.Condition(_queue_lock)
_last_request_at = 0.0
MIN_GAP_S = 0.8  # 两次上游请求最小间隔


class SerialQueue:
    """强制串行 + 最小间隔（worker.js 的 enqueue 语义）。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def run(self, fn):
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_slot)
            self._next_slot = start + MIN_GAP_S
        wait = start - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        return fn()


_queue = SerialQueue()


def _sleep(seconds):
    time.sleep(seconds)


# 解析上游 429/限流响应里的等待时间，返回秒
# 支持格式: "Try again in 2h 51m" / "Try again in 30m" / "Try again in 1h" / "Try again in 15s"
def parse_cooldown(body, status):
    m = re.search(r"try again in (?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?\s*(?:(\d+)\s*s)?", body or "", re.I)
    if m:
        h = int(m.group(1) or 0)
        minutes = int(m.group(2) or 0)
        s = int(m.group(3) or 0)
        seconds = (h * 3600 + minutes * 60 + s)
        if seconds > 0:
            return min(seconds, 6 * 3600)  # 上限 6 小时
    # 429 默认 5 分钟；空响应默认 60 秒
    if status == 429:
        return 5 * 60
    return 60


def cline_fetch_with_retry(path, body_obj, session_id, is_stream=False, max_retries=4):
    """带重试的 clineFetch：429限流/空响应/5xx 自动切换账号 + 指数退避重试。
    所有账号都冷却时，直接返回原始响应（不空转）。"""
    global _current_account
    last_resp = None
    for attempt in range(max_retries + 1):
        # 通过队列串行执行，避免并发空响应
        resp = _queue.run(lambda: cline_fetch(path, body_obj, session_id))
        last_resp = resp

        # 统一读 body
        try:
            body_text = resp.text
        except Exception:  # noqa: BLE001
            body_text = ""

        # 判定"额度/限流"信号（需要切号）：
        # 1. 429（Daily free limit reached / rate limit）
        # 2. 5xx 且含 empty response content
        # 3. 200 非流式但 body 是空响应包装
        is_limit_hit = (
            resp.status_code == 429
            or (resp.status_code >= 500 and "empty response content" in body_text)
            or (resp.status_code < 400 and not is_stream and "empty response content" in body_text)
        )

        if is_limit_hit:
            cooldown_s = parse_cooldown(body_text, resp.status_code)
            acc = _current_account
            if acc is not None:
                acc["cooldownUntil"] = _now_ms() + cooldown_s * 1000
                acc["accessToken"] = None
                acc["expiry"] = 0
                _log(f"[account-switch] 账号额度/限流，冷却 {round(cooldown_s)}s，切换到下一个")
            # 还有可用账号 → 短退避后重试（会切到下一个号）
            pool = parse_accounts()
            has_other = any(not a["cooldownUntil"] or a["cooldownUntil"] <= _now_ms() for a in pool)
            if not has_other:
                _log("[retry] 所有账号均冷却，直接返回上游响应")
                return resp  # 不空转，把 429/错误返回给客户端
            _sleep(0.5 + random.random() * 0.5)
            continue

        # 正常响应（200）或其他错误（403/400/401 等）不重试，直接返回
        return resp
    return last_resp


# ---------------------------------------------------------------------------
# 动态模型列表
# ---------------------------------------------------------------------------

_models_cache = None
_models_cache_time = 0.0
_models_lock = threading.Lock()


def refresh_free_models():
    """Cline 官方免费模型列表（逆向自插件 recommended-models 接口）。"""
    try:
        resp = _client.get(CLINE_API_BASE + "/ai/cline/recommended-models",
                           headers={"User-Agent": "Mozilla/5.0 (cline2api)"})
        if resp.status_code >= 400:
            return []
        data = resp.json()
        free_list = data.get("free") if isinstance(data, dict) else None
        if not isinstance(free_list, list):
            return []
        return [{"id": m["id"], "upstream": m["id"],
                 "provider": (m["id"].split("/")[0] or "cline"), "cost": "free"}
                for m in free_list if isinstance(m, dict) and m.get("id")]
    except Exception:  # noqa: BLE001
        return []


def refresh_models():
    """优先从 Cline 官方 /v1/models 拉取, 失败回退到内置列表。10 分钟缓存。"""
    global _models_cache, _models_cache_time
    now = time.time()
    with _models_lock:
        if _models_cache and now - _models_cache_time < MODELS_TTL:
            return _models_cache
    try:
        resp = _client.get(CLINE_API_BASE + "/models",
                           headers={"User-Agent": "Mozilla/5.0 (cline2api)"})
        if resp.status_code >= 400:
            _log(f"[models] 官方拉取失败 HTTP {resp.status_code} 回退内置列表")
            return FALLBACK_MODELS
        data = resp.json()
        entries = data.get("data") if isinstance(data, dict) else None
        if not isinstance(entries, list) or not entries:
            return FALLBACK_MODELS
        base_list = []
        for m in entries:
            mid = (m or {}).get("id") or ""
            if not mid or mid.endswith(":batch") or m.get("batch"):
                continue
            if ":free" in mid or mid in FREE_WHITELIST:
                prefix = mid.split("/")[0] or "cline"
                base_list.append({"id": mid, "upstream": mid, "provider": prefix, "cost": "free"})
        # 合并 recommended-models 里的 cline-free 免费模型
        for fm in refresh_free_models():
            if not any(b["id"] == fm["id"] for b in base_list):
                base_list.append(fm)
        with _models_lock:
            _models_cache = base_list
            _models_cache_time = now
        _log(f"[models] 动态拉取成功: {len(base_list)} 个模型 (含 cline-free 免费通道)")
        return base_list
    except Exception as exc:  # noqa: BLE001
        _log(f"[models] 拉取异常: {str(exc)[:100]} 回退内置列表")
        return FALLBACK_MODELS


# ---------------------------------------------------------------------------
# OpenAI 协议
# ---------------------------------------------------------------------------


def _json(data, status=200, extra=None):
    headers = {"Content-Type": "application/json", **CORS_HEADERS, **(extra or {})}
    return Response(json.dumps(data), status=status, headers=headers)


def _error_response(message, err_type, status):
    return _json({"error": {"message": message, "type": err_type}}, status)


def get_api_key():
    """API Key 鉴权：未配置 API_KEY → 使用内置默认 key；设为空串则完全关闭鉴权。"""
    provided = os.environ.get("API_KEY")
    if provided == "":
        return "ok"  # 鉴权关闭
    expected = provided if provided else "cline2api-default-key"
    auth = request.headers.get("Authorization") or ""
    if auth.startswith("Bearer "):
        return "ok" if auth[7:] == expected else None
    x_key = request.headers.get("x-api-key")
    return "ok" if x_key == expected else None


def unwrap_data(obj):
    """剥掉上游 {data:{...}} 包装（上游有时包一层 data）。"""
    if isinstance(obj, dict) and isinstance(obj.get("data"), dict):
        d = obj["data"]
        if d.get("choices") or d.get("id") or d.get("usage"):
            return d
    return obj


def handle_chat():
    if get_api_key() is None:
        return _error_response("Invalid API key", "auth_error", 401)
    try:
        params = request.get_json(force=True)
    except Exception:  # noqa: BLE001
        return _error_response("Invalid JSON body", "parse_error", 400)
    if not isinstance(params, dict):
        return _error_response("Invalid JSON body", "parse_error", 400)

    is_stream = bool(params.get("stream"))
    session_id = "sess_" + str(int(time.time() * 1000))
    model = params.get("model") or DEFAULT_MODEL
    model_config = next((m for m in refresh_models() if m["id"] == model), None)
    upstream_model = (model_config or {}).get("upstream") or model

    # 构造上游 body（外部模型 ID 与 Cline 上游模型 ID 分离）
    body = {
        "model": upstream_model,
        "session_id": session_id,
        "reasoning_effort": params.get("reasoning_effort") or params.get("reasoningEffort") or "high",
        "messages": params.get("messages") or [],
    }
    # ⚠️ 上游风控: 免费模型请求体带 max_tokens 字段一律 500 "empty response content"，
    #    剥掉该字段再转发。
    # ⚠️ 免费 DeepSeek 通道：非流式请求被上游限流(500 empty response content)，
    #    流式请求正常。所以客户端要非流式时，强制上游走 stream，再聚合返回。
    force_stream = not is_stream and (
        upstream_model.startswith("deepseek/")
        or upstream_model.startswith("cline-free/")
        or upstream_model.startswith("cline-pass/"))
    if is_stream or force_stream:
        body["stream"] = True
    # 透传可选参数
    for k in ("temperature", "top_p", "tools", "tool_choice", "stop",
              "presence_penalty", "frequency_penalty", "response_format",
              "user", "n", "seed"):
        if params.get(k) is not None:
            body[k] = params[k]

    try:
        resp = cline_fetch_with_retry("/chat/completions", body, session_id, is_stream=True)
        if resp.status_code >= 400:
            return _error_response("upstream error: " + resp.text[:300], "api_error", resp.status_code)
        if is_stream:
            # 客户端要流式：直接透传 SSE（剥 data 包装）
            return stream_response(resp, model)
        if force_stream:
            # 客户端要非流式 + 上游是流式：聚合 chunks 再返回
            retried = non_stream_with_content_check("/chat/completions", body, session_id, resp)
            if retried.get("error"):
                return retried["error"]
            retried["data"]["model"] = model
            return _json(retried["data"], 200)
        # 非流式 + 非 deepseek：原逻辑
        normalized = unwrap_data(resp.json())
        normalized["model"] = model
        return _json(normalized, 200)
    except RuntimeError as exc:
        return _error_response(str(exc), "api_error", 500)


def non_stream_with_content_check(path, body_obj, session_id, first_resp):
    """上游 200 但 content 全空（只有 reasoning）→ 视为坏响应，切号重试。
    最多试 3 次（覆盖多账号切换）。"""
    global _current_account
    max_attempts = 3
    last_data = None
    resp = first_resp
    for attempt in range(max_attempts):
        if resp is None:
            resp = cline_fetch_with_retry(path, body_obj, session_id, is_stream=True)
        if resp.status_code >= 400:
            err_text = resp.text[:300] if resp.text else ""
            return {"error": _error_response("upstream error: " + err_text, "api_error", resp.status_code)}
        ct = resp.headers.get("content-type") or ""
        if "text/event-stream" in ct:
            normalized = stream_to_non_stream(resp)
        else:
            try:
                raw = resp.json()
            except (json.JSONDecodeError, ValueError):
                raw = None
            normalized = unwrap_data(raw) if raw else None
        if not normalized:
            return {"error": _error_response("upstream returned non-SSE body", "api_error", 502)}
        last_data = normalized
        msg = ((normalized.get("choices") or [{}])[0]).get("message") or {}
        content = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning") or "").strip()
        # ⚠️ reasoning 兜底标记：content 为空时 streamToNonStream 会把 reasoning 拼进 content，
        #    这里要识别出来，不能把它当成"好响应"。
        is_reasoning_fallback = msg.get("reasoning_used_as_content") is True
        if content and not is_reasoning_fallback:
            return {"data": normalized}  # 有正式 content → 好响应
        # content 为空（或只有兜底 reasoning）：冷却当前账号并重试
        if reasoning or is_reasoning_fallback:
            acc = _current_account
            if acc is not None:
                acc["cooldownUntil"] = _now_ms() + 30 * 1000  # 短冷却 30s
                acc["accessToken"] = None
                acc["expiry"] = 0
                _log(f"[empty-content] 账号 {attempt} 返回空 content，冷却 30s，重试第 {attempt + 2} 次")
            _sleep(0.3 + random.random() * 0.3)
            resp = None
            continue
        # 完全空（连 reasoning 都没有）→ 也重试
        _log(f"[empty-response] 账号 {attempt} 完全空响应，重试第 {attempt + 2} 次")
        _sleep(0.3 + random.random() * 0.3)
        resp = None
    # 重试用完仍空：返回最后一次（至少带 reasoning，让客户端看到点东西）
    return {"data": last_data}


def stream_to_non_stream(upstream):
    """把上游 SSE 流聚合成 OpenAI 非流式响应对象。
    ⚠️ 兜底：免费通道偶尔整个流只有 reasoning 没有 content（HTTP 200 但空）。
    聚合后发现 content 仍为空且 reasoning 非空时，把 reasoning 拼进 content。"""
    content = ""
    reasoning = ""
    finish_reason = None
    model = ""
    msg_id = ""
    usage = None

    for frame in iter_sse_frames(upstream):
        if not frame.startswith("data:"):
            continue
        payload = frame[5:].strip()
        if payload in ("", "[DONE]"):
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        normalized = unwrap_data(obj)
        choices = (normalized or {}).get("choices")
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        if delta.get("content"):
            content += delta["content"]
        if delta.get("reasoning"):
            reasoning += delta["reasoning"]
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
        if normalized.get("id"):
            msg_id = normalized["id"]
        if normalized.get("model"):
            model = normalized["model"]
        if normalized.get("usage"):
            usage = normalized["usage"]

    msg = {"role": "assistant", "content": content}
    if reasoning:
        msg["reasoning"] = reasoning
    if not content and reasoning:
        msg["content"] = reasoning
        msg["reasoning_used_as_content"] = True
    return {
        "id": msg_id or "gen_" + str(int(time.time() * 1000)),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model or DEFAULT_MODEL,
        "choices": [{
            "index": 0,
            "message": msg,
            "finish_reason": finish_reason or "stop",
            "logprobs": None,
            "native_finish_reason": finish_reason or "stop",
        }],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def iter_sse_frames(resp):
    """Yield complete SSE lines from a streamed httpx response."""
    buf = ""
    for chunk in resp.iter_text():
        buf += chunk
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            yield line


# ---------------------------------------------------------------------------
# Anthropic Messages API → 转 OpenAI 格式再转发
# ---------------------------------------------------------------------------


def handle_anthropic():
    if get_api_key() is None:
        return _error_response("Invalid API key", "auth_error", 401)
    try:
        req = request.get_json(force=True)
    except Exception:  # noqa: BLE001
        return _error_response("Invalid JSON body", "parse_error", 400)
    if not isinstance(req, dict):
        return _error_response("Invalid JSON body", "parse_error", 400)

    is_stream = bool(req.get("stream"))
    session_id = "sess_" + str(int(time.time() * 1000))
    requested_model = req.get("model") or DEFAULT_MODEL
    model_config = next((m for m in refresh_models() if m["id"] == requested_model), None)
    upstream_model = (model_config or {}).get("upstream") or requested_model

    # Anthropic → OpenAI 消息转换
    messages = []
    if req.get("system"):
        sys_content = req["system"] if isinstance(req["system"], str) else json.dumps(req["system"])
        messages.append({"role": "system", "content": sys_content})
    for m in req.get("messages") or []:
        content = m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content"))
        messages.append({"role": m.get("role"), "content": content})

    body = {
        "model": upstream_model,
        "session_id": session_id,
        "reasoning_effort": "high",
        "messages": messages,
    }
    # ⚠️ 上游风控: 免费模型请求体带 max_tokens 字段一律 500，剥离（同 chat/completions 路径）
    # ⚠️ 免费 DeepSeek 通道：非流式被上游限流，强制上游 stream 再聚合
    force_stream = not is_stream and (
        upstream_model.startswith("deepseek/")
        or upstream_model.startswith("cline-free/")
        or upstream_model.startswith("cline-pass/"))
    if is_stream or force_stream:
        body["stream"] = True
    if req.get("temperature") is not None:
        body["temperature"] = req["temperature"]
    if req.get("top_p") is not None:
        body["top_p"] = req["top_p"]
    if req.get("tools"):
        body["tools"] = [{
            "type": "function",
            "function": {"name": t.get("name"), "description": t.get("description") or "",
                         "parameters": t.get("input_schema") or {}},
        } for t in req["tools"] if isinstance(t, dict) and t.get("name")]

    try:
        resp = cline_fetch_with_retry("/chat/completions", body, session_id, is_stream=True)
        if resp.status_code >= 400:
            return _error_response("upstream error: " + resp.text[:300], "api_error", resp.status_code)
        if is_stream:
            # 上游是 OpenAI SSE，转成 Anthropic SSE 格式
            return stream_response_anthropic(resp)
        if force_stream:
            retried = non_stream_with_content_check("/chat/completions", body, session_id, resp)
            if retried.get("error"):
                return retried["error"]
            return _json(openai_to_anthropic(retried["data"]), 200)
        normalized = unwrap_data(resp.json())
        return _json(openai_to_anthropic(normalized), 200)
    except RuntimeError as exc:
        return _error_response(str(exc), "api_error", 500)


# ---------------------------------------------------------------------------
# 响应处理（SSE 流）
# ---------------------------------------------------------------------------


def stream_response(upstream, external_model):
    """OpenAI SSE 流式透传（剥 data 包装）。"""

    def generate():
        buf = ""
        for chunk in upstream.iter_text():
            buf += chunk
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                if line.startswith("data:"):
                    payload = line[5:].strip()
                    if payload in ("", "[DONE]"):
                        yield line + "\n\n"
                        continue
                    try:
                        obj = json.loads(payload)
                        normalized = unwrap_data(obj)
                        if normalized and external_model:
                            normalized["model"] = external_model
                        yield "data: " + json.dumps(normalized) + "\n\n"
                    except json.JSONDecodeError:
                        yield line + "\n"
                else:
                    yield line + "\n"

    return Response(generate(), status=200, headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        **CORS_HEADERS,
    })


def stream_response_anthropic(upstream):
    """Anthropic SSE：把上游 OpenAI chunk 转成 Anthropic 格式。"""

    def generate():
        buf = ""
        emitted = False
        for chunk in upstream.iter_text():
            buf += chunk
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload in ("", "[DONE]"):
                    continue
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                normalized = unwrap_data(obj)
                choices = (normalized or {}).get("choices")
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    emitted = True
                    yield ("event: content_block_delta\ndata: "
                           + json.dumps({"type": "content_block_delta", "index": 0,
                                         "delta": {"type": "text_delta", "text": delta["content"]}})
                           + "\n\n")
                tool_calls = delta.get("tool_calls") or []
                for tc in tool_calls:
                    partial = (tc.get("function") or {}).get("arguments") or ""
                    emitted = True
                    yield ("event: content_block_delta\ndata: "
                           + json.dumps({"type": "content_block_delta", "index": 0,
                                         "delta": {"type": "input_json_delta", "partial_json": partial}})
                           + "\n\n")
        # 结束事件
        yield ("event: message_delta\ndata: "
               + json.dumps({"type": "message_delta",
                             "delta": {"stop_reason": "end_turn"},
                             "usage": {"output_tokens": 0}})
               + "\n\n")
        yield ("event: message_stop\ndata: "
               + json.dumps({"type": "message_stop"})
               + "\n\n")

    return Response(generate(), status=200, headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        **CORS_HEADERS,
    })


def openai_to_anthropic(openai_resp):
    choice = ((openai_resp or {}).get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    return {
        "id": (openai_resp or {}).get("id") or "msg_" + str(int(time.time() * 1000)),
        "type": "message",
        "role": "assistant",
        "model": (openai_resp or {}).get("model") or "",
        "content": [{"type": "text", "text": content}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": ((openai_resp or {}).get("usage") or {}).get("prompt_tokens") or 0,
            "output_tokens": ((openai_resp or {}).get("usage") or {}).get("completion_tokens") or 0,
        },
    }


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------


@app.route("/v1/health", methods=["GET"])
def health():
    return _json({
        "ok": True,
        "version": VERSION,
        "authenticated": bool(os.environ.get("API_KEY")),
        "accounts": len(parse_accounts()),
        "model": DEFAULT_MODEL,
    }, 200)


@app.route("/v1/models", methods=["GET"])
@app.route("/models", methods=["GET"])
def models():
    # GET /v1/models — 免鉴权（GUI 验证需拉模型列表）
    lst = refresh_models()
    payload = [{"id": m["id"], "object": "model",
                "created": int(time.time()), "owned_by": "cline"} for m in lst]
    return _json({"object": "list", "data": payload}, 200, {"X-Cline2api-Version": VERSION})


@app.before_request
def _options_short_circuit():
    if request.method == "OPTIONS":
        return Response(status=204, headers=CORS_HEADERS)


@app.route("/v1/chat/completions", methods=["POST"])
@app.route("/chat/completions", methods=["POST"])
def chat():
    return handle_chat()


@app.route("/v1/messages", methods=["POST"])
@app.route("/messages", methods=["POST"])
def messages():
    return handle_anthropic()


@app.route("/<path:_any>", methods=["GET", "POST"])
def not_found(_any):
    return _json({"error": {"message": "Not found", "type": "not_found"}}, 404)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    host = os.environ.get("HOST", "0.0.0.0")
    app.run(host=host, port=port, threaded=True)
