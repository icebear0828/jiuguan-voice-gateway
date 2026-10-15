"""
Unit tests for core/voice_router.py
Testing emotion tag parsing, heuristic inference, parameter inheritance hierarchy, and fallback mechanisms.
"""

import pytest
from unittest.mock import patch, MagicMock
from core.voice_router import VoiceRouter, VoiceSynthesisPlan


@pytest.mark.unit
class TestEmotionTagExtraction:
    def test_square_brackets_tag(self):
        text = "[playful] 早上好呀，今天天气真不错！"
        clean, emo = VoiceRouter.extract_emotion_tag(text)
        assert emo == "playful"
        assert clean == "早上好呀，今天天气真不错！"

    def test_chinese_brackets_tag(self):
        text = "【威严】 此乃吾之领地，退后！"
        clean, emo = VoiceRouter.extract_emotion_tag(text)
        assert emo == "confident"
        assert clean == "此乃吾之领地，退后！"

    def test_xml_tag_extraction(self):
        text = '<emotion name="tsundere">哼，我才不是特意为你准备的！</emotion>'
        clean, emo = VoiceRouter.extract_emotion_tag(text)
        assert emo == "tsundere"
        assert clean == "哼，我才不是特意为你准备的！"

    def test_parentheses_tag(self):
        text = "(害羞) 那个...其实我也有一点点喜欢你。"
        clean, emo = VoiceRouter.extract_emotion_tag(text)
        assert emo == "shy"
        assert clean == "那个...其实我也有一点点喜欢你。"

    def test_fullwidth_parentheses_tag(self):
        text = "（傲娇）笨蛋，不要误会了！"
        clean, emo = VoiceRouter.extract_emotion_tag(text)
        assert emo == "tsundere"
        assert clean == "笨蛋，不要误会了！"

    def test_no_emotion_tag(self):
        text = "普通的日常对话，没有任何标签。"
        clean, emo = VoiceRouter.extract_emotion_tag(text)
        assert emo is None
        assert clean == text


@pytest.mark.unit
class TestHeuristicClassification:
    def test_rimuru_keywords(self):
        # "悪いスライムじゃないよ" should trigger playful
        emo = VoiceRouter.classify_by_heuristics("rimuru", "ぼくは悪いスライムじゃないよ！")
        assert emo == "playful"

        # "魔王" should trigger confident
        emo = VoiceRouter.classify_by_heuristics("rimuru", "我即是八星魔王之一！")
        assert emo == "confident"

        # "温泉" should trigger slacking
        emo = VoiceRouter.classify_by_heuristics("rimuru", "今天好累，一起去泡温泉吧。")
        assert emo == "slacking"

    def test_kurisu_keywords(self):
        # "変態" should trigger tsundere
        emo = VoiceRouter.classify_by_heuristics("kurisu", "あんた、何よこの変態！")
        assert emo == "tsundere"

        # "タイムマシン" should trigger science
        emo = VoiceRouter.classify_by_heuristics("kurisu", "タイムマシンについての理論的仮説を説明するわ。")
        assert emo == "science"

    def test_punctuation_heuristics(self):
        # Exclamation marks for rimuru -> confident
        emo = VoiceRouter.classify_by_heuristics("rimuru", "覚悟しなさい！！")
        assert emo == "confident"

        # Exclamation marks for kurisu -> tsundere
        emo = VoiceRouter.classify_by_heuristics("kurisu", "バカ！？")
        assert emo == "tsundere"

        # Ellipsis -> gentle
        emo = VoiceRouter.classify_by_heuristics("rimuru", "そんなこと言われても……")
        assert emo == "gentle"


@pytest.mark.unit
class TestParameterInheritancePriority:
    def test_user_params_highest_priority(self):
        """User parameter overrides character and default values."""
        plan = VoiceRouter.resolve_plan(
            character="rimuru",
            text="你好呀",
            explicit_emotion="playful",
            user_params={"temperature": 0.42, "speed": 1.35, "top_k": 7, "top_p": 0.88}
        )
        assert isinstance(plan, VoiceSynthesisPlan)
        assert plan.temperature == 0.42
        assert plan.speed_factor == 1.35
        assert plan.top_k == 7
        assert plan.top_p == 0.88
        assert "explicit_param" in plan.debug_source

    def test_explicit_emotion_overrides_inline_tag(self):
        """Explicit emotion passed in API takes priority over inline [tag]."""
        plan = VoiceRouter.resolve_plan(
            character="rimuru",
            text="[slacking] 我想睡午觉了",
            explicit_emotion="confident"
        )
        assert plan.emotion_id == "confident"
        assert "explicit_param" in plan.debug_source

    def test_inline_tag_overrides_heuristics(self):
        """Inline tag overrides heuristic classification."""
        # Contains heuristic "変態" (tsundere), but explicit tag is [science]
        plan = VoiceRouter.resolve_plan(
            character="kurisu",
            text="[science] 変態的仮説について検証しよう。"
        )
        assert plan.emotion_id == "science"
        assert "inline_tag" in plan.debug_source
        assert "[science]" not in plan.clean_text

    def test_fallback_when_reference_audio_missing(self):
        """If target emotion audio doesn't exist, safely falls back to default reference audio."""
        with patch.object(VoiceRouter, "get_available_emotions") as mock_get_emotions:
            mock_get_emotions.return_value = [
                {
                    "id": "daily",
                    "name": "default",
                    "emotion": "日常",
                    "is_default": True,
                    "audio_path": "/fake/path/default.wav",
                    "text": "默认提示词",
                    "lang": "ja",
                    "recommended_params": {"temperature": 0.8}
                },
                {
                    "id": "special_rare",
                    "name": "rare",
                    "emotion": "稀有",
                    "is_default": False,
                    "audio_path": "/fake/nonexistent/rare.wav",
                    "text": "稀有提示词",
                    "lang": "ja",
                    "recommended_params": {"temperature": 0.6}
                }
            ]

            plan = VoiceRouter.resolve_plan(
                character="kurisu",
                text="测试缺失音频回退",
                explicit_emotion="special_rare"
            )
            # Should safely fallback to default audio
            assert plan.is_fallback is True
            assert plan.ref_audio_path == "/fake/path/default.wav"


@pytest.mark.unit
class TestEnsureCharacterWeights:
    def test_already_loaded_cache(self):
        VoiceRouter._loaded_weights["character"] = "rimuru"
        assert VoiceRouter.ensure_character_weights("rimuru") is True

    @patch("http.client.HTTPConnection")
    def test_successful_switch(self, mock_conn_cls):
        VoiceRouter._loaded_weights["character"] = None
        mock_conn = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"ok"
        mock_conn.getresponse.return_value = mock_resp
        mock_conn_cls.return_value = mock_conn

        # Mock existence of model weights
        with patch("os.path.exists", return_value=True):
            res = VoiceRouter.ensure_character_weights("kurisu", base_url="http://127.0.0.1:9880")
            assert res is True
            assert VoiceRouter._loaded_weights["character"] == "kurisu"
            assert mock_conn.request.call_count == 2
