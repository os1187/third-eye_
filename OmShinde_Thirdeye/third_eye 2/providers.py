import os
import time
from typing import Optional, Dict, Any


def _safe_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


class TTSProviderBase:
    def synthesize(self, text: str) -> Optional[bytes]:
        raise NotImplementedError


class ElevenLabsTTS(TTSProviderBase):
    def __init__(self, voice_id: str, api_key: Optional[str] = None, timeout_s: float = 3.0):
        self.voice_id = voice_id or ""
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY", "")
        self.timeout_s = float(timeout_s)
        self._requests = _safe_import('requests')

    def synthesize(self, text: str) -> Optional[bytes]:
        if not self._requests or not self.api_key or not self.voice_id:
            return None
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream"
            headers = {
                "xi-api-key": self.api_key,
                "accept": "audio/mpeg",
                "content-type": "application/json"
            }
            payload = {
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}
            }
            r = self._requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
            r.raise_for_status()
            return r.content
        except Exception:
            return None


class PhrasingProviderBase:
    def phrase(self, context: Dict[str, Any]) -> Optional[str]:
        raise NotImplementedError


class OpenAIPhraser(PhrasingProviderBase):
    def __init__(self, timeout_s: float = 2.0):
        self.timeout_s = float(timeout_s)
        self._openai = _safe_import('openai')
        api_key = os.getenv('OPENAI_API_KEY', '')
        if self._openai and api_key:
            try:
                self._openai.api_key = api_key
            except Exception:
                pass

    def phrase(self, context: Dict[str, Any]) -> Optional[str]:
        if not self._openai or not getattr(self._openai, 'api_key', None):
            return None
        try:
            # Keep prompt compact to reduce latency
            hazards = context.get('labels', [])
            sides = context.get('sides', [])
            urgency = context.get('urgency', 'info')
            corridor = context.get('corridor_hint', '')
            raw = context.get('raw', '')
            travel = context.get('travel_mode', 'WALKING')
            prompt = (
                "You are a safety announcer. Keep it concise and actionable.\n"
                f"Urgency: {urgency}. Travel: {travel}. Corridor: {corridor}.\n"
                f"Hazards: {', '.join(map(str, hazards))}. Sides: {', '.join(map(str, sides))}.\n"
                f"Base: {raw[:140]}\n"
                "Output a short phrase including side and distance if present."
            )
            # Use chat completion if available, otherwise text completion
            try:
                resp = self._openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Safety, concise, avoid filler."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    timeout=self.timeout_s,
                )
                text = resp.choices[0].message.content.strip()
            except Exception:
                # Fallback: return raw limited
                text = raw
            return text[:160]
        except Exception:
            return None


def get_tts_provider(config: Dict[str, Any]) -> Optional[TTSProviderBase]:
    prov = str(config.get('audio', {}).get('tts_provider', 'local') or 'local').lower()
    timeout_s = float(config.get('audio', {}).get('tts_timeout_s', 3.0))
    if prov == 'elevenlabs':
        voice_id = str(config.get('audio', {}).get('voice_id', '') or '')
        return ElevenLabsTTS(voice_id=voice_id, timeout_s=timeout_s)
    # TODO: add azure, google, piper, coqui adapters
    return None


def get_phrasing_provider(config: Dict[str, Any]) -> Optional[PhrasingProviderBase]:
    ann = config.get('announcements', {})
    if not bool(ann.get('phrasing_enabled', False)):
        return None
    prov = str(ann.get('phrasing_provider', 'builtin') or 'builtin').lower()
    timeout_s = float(ann.get('phrasing_timeout_s', 2.0))
    if prov == 'openai':
        return OpenAIPhraser(timeout_s=timeout_s)
    # TODO: anthropic, local_transformer
    return None