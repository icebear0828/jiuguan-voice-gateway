#!/usr/bin/env python3
"""
VoiceRouter: Intelligent Multi-Emotion Voice Routing & Prompt Conditioning Engine
Handles dynamic emotion extraction, prompt audio resolution, heuristic classification,
and parameter optimization for GPT-SoVITS anime characters.
"""

import os
import re
import json
import urllib.parse
import http.client
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "assets/models")
REGISTRY_PATH = os.path.join(MODELS_DIR, "registry.json")

# Character-specific emotion heuristic keywords
HEURISTIC_KEYWORDS = {
    "rimuru": {
        "playful": ["スライム", "悪いスライムじゃないよ", "わはは", "ワハハ", "えへへ", "～☆", "～♪", "どうだ", "ふふん", "美味い"],
        "confident": ["後悔させてやる", "動くなよ", "敵", "戦", "屈服", "我こそ", "命じる", "逃がさん", "覚悟", "貴様", "魔王", "誓う"],
        "gentle": ["約束", "守る", "安心", "大丈夫", "優しい", "ありがとう", "仲間たち", "笑顔", "安らかに", "暮らせる国"],
        "slacking": ["サボ", "温泉", "めんどくさい", "ふぁ〜あ", "シオンの料理", "いやいや", "パスで", "昼寝", "だるい"]
    },
    "kurisu": {
        "tsundere": ["変態", "バカ", "死ぬの", "何よ", "勘違いしないで", "別に", "あんた", "どアホ", "フン", "うるさい"],
        "science": ["理論", "科学", "仮説", "実験", "論文", "タイムマシン", "量子", "脳科学", "因果律", "世界線"],
        "gentle": ["岡部", "辛そう", "泣き出しそう", "大丈夫", "側にいる", "ありがとう", "忘れないで", "ずっと"]
    },
    "utaha": {
        "annoyed": ["毒虫", "締め切り", "白紙", "プロット", "何よ", "怒", "どうして"],
        "intimate": ["倫理君", "ふふっ", "見つめて", "温もり", "重み", "耳元", "そばに"],
        "shy": ["勘違い", "別に", "激しく", "優しく", "恥ずかしい"],
        "sad": ["さようなら", "初恋", "戻れない", "泣", "辛い", "悲しい"],
        "neutral": ["小説", "作家", "物語", "義務", "特権", "原稿"]
    },
    "chizuru": {
        "gentle": ["彼女", "私", "デート", "大事", "時間", "またね", "契約"],
        "intimate": ["伸び伸び", "期待", "ふふ", "一緒", "手をつな", "二人"],
        "tsundere": ["別に", "勘違い", "言ってない", "バカ", "何よ", "あなたと"],
        "shy": ["ドキドキ", "久しぶり", "恥ずかしい", "赤く", "顔"],
        "annoyed": ["大した用", "速攻", "切る", "もう", "何なの"],
        "excited": ["見て見て", "ペンギン", "わー", "すごい", "かわいい", "きゃー"],
        "happy": ["嬉しい", "喜んで", "よかった", "ありがとう", "笑顔"],
        "sad": ["寂しく", "夕焼け", "さようなら", "終わり", "切ない"],
        "angry": ["夜に干す", "本当だ", "怒", "もう！"]
    }
}

TAG_NORMALIZATION_MAP = {
    # Playful
    "playful": "playful", "cute": "playful", "俏皮": "playful", "卖萌": "playful", "开心": "playful", "得意": "playful", "活泼": "playful",
    # Confident / Combat
    "confident": "confident", "combat": "confident", "battle": "confident", "angry": "confident", "威严": "confident", "霸气": "confident", "战斗": "confident", "愤怒": "confident", "严肃": "confident", "命令": "confident",
    # Gentle / Intimate
    "gentle": "gentle", "tender": "gentle", "promise": "gentle", "温柔": "gentle", "深情": "gentle", "治愈": "gentle", "承诺": "gentle", "感动": "gentle", "心疼": "gentle",
    "intimate": "intimate", "亲密": "intimate", "耳语": "intimate", "挑逗": "intimate", "暧昧": "intimate",
    # Slacking
    "slacking": "slacking", "humor": "slacking", "funny": "slacking", "lazy": "slacking", "吐槽": "slacking", "摸鱼": "slacking", "搞笑": "slacking", "无奈": "slacking", "怠惰": "slacking",
    # Tsundere / Shy / Annoyed
    "tsundere": "tsundere", "傲娇": "tsundere", "怒斥": "tsundere",
    "shy": "shy", "害羞": "shy", "微嗔": "shy", "脸红": "shy",
    "annoyed": "annoyed", "不满": "annoyed", "斥责": "annoyed", "嗔怒": "annoyed", "毒舌": "annoyed",
    # Sad
    "sad": "sad", "sorrow": "sad", "悲伤": "sad", "哀伤": "sad", "哭泣": "sad", "失落": "sad",
    # Science / Neutral
    "science": "science", "知性": "science", "科研": "science", "理性": "science",
    "neutral": "neutral", "中立": "neutral", "独白": "neutral",
    # Daily / Fallback
    "daily": "daily", "general": "daily", "日常": "daily", "平静": "daily", "自介": "daily", "普通": "daily"
}


@dataclass
class VoiceSynthesisPlan:
    character_id: str
    emotion_id: str
    emotion_label: str
    clean_text: str
    ref_audio_path: str
    prompt_text: str
    prompt_lang: str
    temperature: float
    speed_factor: float
    top_k: int
    top_p: int
    is_fallback: bool
    debug_source: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VoiceRouter:
    _loaded_weights = {"character": None}

    @classmethod
    def get_character_model_dir(cls, character: str) -> str:
        if not os.path.exists(REGISTRY_PATH):
            return os.path.join(MODELS_DIR, character, "v2pro_master")
        try:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                reg = json.load(f)
            char_data = reg.get("characters", {}).get(character, {})
            active_ver = char_data.get("active_version")
            if active_ver and active_ver in char_data.get("versions", {}):
                rel_dir = char_data["versions"][active_ver]["model_dir"]
                return os.path.join(MODELS_DIR, rel_dir)
            if active_ver and os.path.isdir(os.path.join(MODELS_DIR, character, active_ver)):
                return os.path.join(MODELS_DIR, character, active_ver)
            if os.path.exists(os.path.join(MODELS_DIR, character, "current")):
                return os.path.join(MODELS_DIR, character, "current")
        except Exception:
            pass
        return os.path.join(MODELS_DIR, character, "v2pro_master")

    @classmethod
    def get_available_emotions(cls, character: str) -> List[Dict[str, Any]]:
        """Returns all registered emotions and status for a character."""
        model_dir = cls.get_character_model_dir(character)
        
        # Check prompt_metadata.json in reference_audio(s)
        candidates = [
            os.path.join(model_dir, "reference_audio", "prompt_metadata.json"),
            os.path.join(model_dir, "reference_audios", "prompt_metadata.json"),
            os.path.join(model_dir, "model_info.json")
        ]
        
        for cand in candidates:
            if not os.path.exists(cand):
                continue
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Handle model_info.json style
                if isinstance(data, dict) and "reference_audios" in data:
                    data = data["reference_audios"]
                
                if isinstance(data, list):
                    res = []
                    for item in data:
                        eid = item.get("id") or item.get("category", "daily")
                        rel_file = item.get("file") or item.get("audio_file", "")
                        full_file = os.path.join(model_dir, rel_file) if not os.path.isabs(rel_file) else rel_file
                        exists = os.path.exists(full_file)
                        res.append({
                            "id": eid,
                            "name": item.get("name", eid),
                            "emotion": item.get("emotion", eid),
                            "text": item.get("text") or item.get("prompt_text", ""),
                            "lang": item.get("lang") or item.get("prompt_lang", "ja"),
                            "is_default": bool(item.get("is_default", False)),
                            "recommended_params": item.get("recommended_params", {}),
                            "file_exists": exists,
                            "audio_path": full_file
                        })
                    return res
            except Exception as e:
                print(f"[VoiceRouter] Error loading emotion metadata from {cand}: {e}")

        # Fallback default
        return [{
            "id": "daily",
            "name": "default",
            "emotion": "日常默认",
            "text": "",
            "lang": "ja",
            "is_default": True,
            "recommended_params": {},
            "file_exists": True,
            "audio_path": ""
        }]

    @classmethod
    def extract_emotion_tag(cls, text: str) -> Tuple[str, Optional[str]]:
        """
        Extracts emotion tag (e.g. [playful], (开心), 【威严】) and returns (clean_text, matched_emotion).
        """
        # Pattern 1: [tag] or 【tag】
        m = re.search(r"[\[【]([a-zA-Z0-9_\u4e00-\u9fa5]+)[\]】]", text)
        if m:
            tag_raw = m.group(1).strip().lower()
            if tag_raw in TAG_NORMALIZATION_MAP:
                norm = TAG_NORMALIZATION_MAP[tag_raw]
                clean = text[:m.start()] + text[m.end():]
                return clean.strip(), norm
            # Chinese character lookup
            for k, v in TAG_NORMALIZATION_MAP.items():
                if k in tag_raw:
                    clean = text[:m.start()] + text[m.end():]
                    return clean.strip(), v

        # Pattern 2: (tag) or （tag）
        m = re.search(r"[\(（]([a-zA-Z0-9_\u4e00-\u9fa5]+)[\)）]", text)
        if m:
            tag_raw = m.group(1).strip().lower()
            if tag_raw in TAG_NORMALIZATION_MAP:
                norm = TAG_NORMALIZATION_MAP[tag_raw]
                clean = text[:m.start()] + text[m.end():]
                return clean.strip(), norm
            for k, v in TAG_NORMALIZATION_MAP.items():
                if k in tag_raw:
                    clean = text[:m.start()] + text[m.end():]
                    return clean.strip(), v

        # Pattern 3: <emotion name="...">
        m = re.search(r"<emotion(?:\s+name)?=[\"']?([a-zA-Z0-9_]+)[\"']?>", text, re.IGNORECASE)
        if m:
            tag_raw = m.group(1).strip().lower()
            norm = TAG_NORMALIZATION_MAP.get(tag_raw, tag_raw)
            clean = re.sub(r"</?emotion[^>]*>", "", text)
            return clean.strip(), norm

        return text, None

    @classmethod
    def classify_by_heuristics(cls, character: str, text: str) -> Optional[str]:
        """Classifies text emotion by character-specific keywords and punctuation."""
        kw_map = HEURISTIC_KEYWORDS.get(character, {})
        for emo_id, kws in kw_map.items():
            if any(kw in text for kw in kws):
                return emo_id
        
        # Punctuation heuristics
        if "！？" in text or "!!" in text or "！！" in text:
            if character == "rimuru":
                return "confident"
            elif character == "kurisu":
                return "tsundere"
        if "……" in text or "..." in text:
            return "gentle"

        return None

    @classmethod
    def resolve_plan(
        cls,
        character: str,
        text: str,
        explicit_emotion: Optional[str] = None,
        user_params: Optional[Dict[str, Any]] = None
    ) -> VoiceSynthesisPlan:
        """
        Resolves the comprehensive synthesis plan (audio, text, params) based on priority:
        1. Explicit parameter
        2. Inline text emotion tag
        3. Heuristic keyword matching
        4. Default character emotion
        """
        user_params = user_params or {}
        clean_text = text.strip()
        matched_emotion = None
        source = "default"

        # 1. Explicit emotion
        if explicit_emotion:
            norm_emo = TAG_NORMALIZATION_MAP.get(explicit_emotion.lower(), explicit_emotion.lower())
            matched_emotion = norm_emo
            source = f"explicit_param({explicit_emotion})"

        # 2. Inline tag
        if not matched_emotion:
            clean_text, tag_emo = cls.extract_emotion_tag(clean_text)
            if tag_emo:
                matched_emotion = tag_emo
                source = f"inline_tag({tag_emo})"

        # 3. Heuristic rule
        if not matched_emotion:
            rule_emo = cls.classify_by_heuristics(character, clean_text)
            if rule_emo:
                matched_emotion = rule_emo
                source = f"heuristic_rule({rule_emo})"

        # 4. Fallback to default
        if not matched_emotion:
            if character == "utaha":
                matched_emotion = "neutral"
            elif character == "kurisu":
                matched_emotion = "tsundere"
            elif character == "yutao":
                matched_emotion = "default"
            else:
                matched_emotion = "daily"
            source = "default_fallback"

        # Load character emotions
        emotions = cls.get_available_emotions(character)
        model_dir = cls.get_character_model_dir(character)

        # Find target emotion entry
        target_entry = None
        default_entry = None
        for emo in emotions:
            if emo["is_default"]:
                default_entry = emo
            if emo["id"] == matched_emotion:
                target_entry = emo

        if not default_entry and emotions:
            default_entry = emotions[0]
        if not target_entry:
            target_entry = default_entry

        # Resolve audio file and prompt text
        is_fallback = False
        target_audio = target_entry.get("audio_path", "")
        target_prompt = target_entry.get("text", "")
        target_lang = target_entry.get("lang", "ja")

        if not os.path.exists(target_audio):
            # Fallback to default audio for acoustic safety
            is_fallback = True
            target_audio = default_entry.get("audio_path", "")
            target_prompt = default_entry.get("text", "")
            target_lang = default_entry.get("lang", "ja")

        # Resolve params (hierarchy: user override > emotion recommended > global default)
        rec = target_entry.get("recommended_params", {})
        default_temp = 0.70 if character == "rimuru" else 1.0
        default_speed = 1.0

        def _val(k1, k2, fallback):
            v1 = user_params.get(k1)
            if v1 is not None:
                return v1
            if k2:
                v2 = user_params.get(k2)
                if v2 is not None:
                    return v2
            vr = rec.get(k1)
            if vr is not None:
                return vr
            return fallback

        temperature = float(_val("temperature", None, default_temp))
        speed_factor = float(_val("speed_factor", "speed", default_speed))
        top_k = int(_val("top_k", None, 5))
        top_p = float(_val("top_p", None, 0.95 if character == "rimuru" else 1.0))

        return VoiceSynthesisPlan(
            character_id=character,
            emotion_id=target_entry["id"],
            emotion_label=target_entry.get("emotion", target_entry["id"]),
            clean_text=clean_text,
            ref_audio_path=target_audio,
            prompt_text=target_prompt,
            prompt_lang=target_lang,
            temperature=temperature,
            speed_factor=speed_factor,
            top_k=top_k,
            top_p=top_p,
            is_fallback=is_fallback,
            debug_source=source
        )

    @classmethod
    def ensure_character_weights(cls, character: str, base_url: str = "http://127.0.0.1:9880") -> bool:
        """Hot-swaps GPT-SoVITS character weights on port 9880 if not already loaded."""
        if cls._loaded_weights.get("character") == character:
            return True

        model_dir = cls.get_character_model_dir(character)
        info_path = os.path.join(model_dir, "model_info.json")
        gpt_path = ""
        sovits_path = ""

        if os.path.exists(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    minfo = json.load(f)
                weights = minfo.get("weights", {})
                if "gpt" in weights:
                    gpt_path = os.path.join(model_dir, weights["gpt"])
                if "sovits" in weights:
                    sovits_path = os.path.join(model_dir, weights["sovits"])
            except Exception as e:
                print(f"[VoiceRouter] Error reading {info_path}: {e}")

        # Fallback known paths
        if not gpt_path or not sovits_path:
            if character == "kurisu":
                gpt_path = os.path.join(MODELS_DIR, "kurisu/v2pro_master/Kurisu-e20.ckpt")
                sovits_path = os.path.join(MODELS_DIR, "kurisu/v2pro_master/Kurisu_e25_s9475.pth")
            elif character == "rimuru":
                gpt_path = os.path.join(MODELS_DIR, "rimuru/v2pro_master/Rimuru_v2Pro_Native-e10.ckpt")
                sovits_path = os.path.join(MODELS_DIR, "rimuru/v2pro_master/Rimuru_v2Pro_e25_s6825.pth")
            elif character == "mika":
                gpt_path = os.path.join(MODELS_DIR, "mika/mika-e15.ckpt")
                sovits_path = os.path.join(MODELS_DIR, "mika/mika_e8_s176.pth")
            elif character == "utaha":
                gpt_path = os.path.join(MODELS_DIR, "utaha/v4/霞之丘诗羽_01-e10.ckpt")
                sovits_path = os.path.join(MODELS_DIR, "utaha/v4/霞之丘诗羽_01_e16_s368_l32.pth")
            elif character == "chizuru":
                gpt_path = os.path.join(MODELS_DIR, "chizuru/v4/水原千鹤-v4dpo-20260607-e10.ckpt")
                sovits_path = os.path.join(MODELS_DIR, "chizuru/v4/水原千鹤-v4dpo-20260607_e16_s464_l32.pth")
            elif character == "yutao":
                gpt_path = os.path.join(MODELS_DIR, "yutao/v2pro_master/my_voice_private-e4.ckpt")
                sovits_path = os.path.join(MODELS_DIR, "yutao/v2pro_master/my_voice_private_e4_s76.pth")

        if not (os.path.exists(gpt_path) and os.path.exists(sovits_path)):
            print(f"[VoiceRouter] Missing model weights for {character}: GPT={gpt_path}, SoVITS={sovits_path}")
            return False

        try:
            parsed = urllib.parse.urlparse(base_url)
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or 9880
            conn = http.client.HTTPConnection(host, port, timeout=10)
            
            conn.request("GET", f"/set_gpt_weights?weights_path={urllib.parse.quote(gpt_path)}")
            r1 = conn.getresponse()
            r1.read()
            
            conn.request("GET", f"/set_sovits_weights?weights_path={urllib.parse.quote(sovits_path)}")
            r2 = conn.getresponse()
            r2.read()
            
            conn.close()
            cls._loaded_weights["character"] = character
            print(f"✅ [VoiceRouter] 成功热加载角色权重: {character}")
            return True
        except Exception as e:
            print(f"⚠️ [VoiceRouter] 权重热切换异常: {e}")
            return False

