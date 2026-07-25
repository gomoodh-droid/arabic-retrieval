"""
عميل نماذج محايد للمزوّد: Anthropic / OpenAI-compatible / Ollama.
وإن لم يوجد مفتاح، يعمل المشروع في وضع offline بمنطق استكشافي —
حتى يستطيع أي أحد تشغيل الحلقة كاملة في خمس دقائق بلا حساب ولا مفتاح.
"""
from __future__ import annotations
import json, os, re, urllib.request

class LLM:
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = provider or os.getenv("ARL_PROVIDER", "offline")
        self.model = model or os.getenv("ARL_MODEL", "")
        self.calls = 0

    @property
    def online(self) -> bool:
        return self.provider != "offline"

    def ask(self, system: str, user: str, max_tokens: int = 1200) -> str:
        """يرجع نصًا. لا يرمي استثناءً — يرجع '' عند الفشل ليتولى الوكيل بديله."""
        if not self.online:
            return ""
        self.calls += 1
        try:
            if self.provider == "anthropic":
                return self._post(
                    "https://api.anthropic.com/v1/messages",
                    {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
                    {"model": self.model or "claude-sonnet-4-6", "max_tokens": max_tokens,
                     "system": system, "messages": [{"role": "user", "content": user}]},
                    lambda d: "".join(b.get("text", "") for b in d["content"]))
            if self.provider == "ollama":
                base = os.getenv("OLLAMA_HOST", "http://localhost:11434")
                return self._post(f"{base}/api/chat", {},
                    {"model": self.model or "qwen3", "stream": False,
                     "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                    lambda d: d["message"]["content"])
            # openai-compatible (OpenAI, DeepSeek, GLM, Qwen, Together, vLLM ...)
            base = os.getenv("ARL_BASE_URL", "https://api.openai.com/v1")
            return self._post(f"{base}/chat/completions",
                {"Authorization": f"Bearer {os.environ.get('ARL_API_KEY', '')}"},
                {"model": self.model or "gpt-4.1-mini", "max_tokens": max_tokens,
                 "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                lambda d: d["choices"][0]["message"]["content"])
        except Exception as e:                      # noqa: BLE001
            print(f"  [llm] تعذّر الاستدعاء ({e}) — الرجوع إلى الوضع الاستكشافي")
            return ""

    @staticmethod
    def _post(url, headers, payload, extract):
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"content-type": "application/json", **headers})
        with urllib.request.urlopen(req, timeout=120) as r:
            return extract(json.loads(r.read()))


def parse_json(text: str, default):
    """النماذج تُغلّف JSON بأسوار وشرح — نستخرجه بمرونة."""
    if not text:
        return default
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    for cand in (text, *re.findall(r"[\[{].*[\]}]", text, re.S)):
        try:
            return json.loads(cand)
        except Exception:  # noqa: BLE001
            continue
    return default
