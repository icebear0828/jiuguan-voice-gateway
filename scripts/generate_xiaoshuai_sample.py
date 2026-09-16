#!/usr/bin/env python3
"""
Authentic 'Xiaoshuai' Movie Narration Voice Generator (Microsoft Yunxi Neural)
Generates the iconic commercial-standard Chinese movie narration clips.

Usage:
    python scripts/generate_xiaoshuai_sample.py
"""

import asyncio
import os
import edge_tts

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "assets/audio/xiaoshuai_samples")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLES = [
    {
        "id": "xiaoshuai_01_classic_intro",
        "title": "经典开场白",
        "text": "注意看，眼前这个男人叫小帅，他怎么也没想到，自己竟然穿越到了一个完全陌生的高武世界！",
        "voice": "zh-CN-YunxiNeural",
        "rate": "+10%",
        "pitch": "+0Hz"
    },
    {
        "id": "xiaoshuai_02_suspense_plot",
        "title": "剧情转折与悬疑",
        "text": "就在所有人都以为他要彻底放弃的时候，小帅的嘴角却微微上扬，因为真正的反击，才刚刚开始！",
        "voice": "zh-CN-YunxiNeural",
        "rate": "+15%",
        "pitch": "+0Hz"
    },
    {
        "id": "xiaoshuai_03_knowledge_explain",
        "title": "知识讲解与科普",
        "text": "其实量子力学并没有大家想象的那么玄乎，今天我就用三分钟的时间，带大家彻底搞懂什么是薛定谔的猫。",
        "voice": "zh-CN-YunxiNeural",
        "rate": "+5%",
        "pitch": "+0Hz"
    }
]


async def generate_all():
    print("=" * 70)
    print("🎬 正宗影视解说【小帅】（云希·解说音色）成品批量生成器")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print("=" * 70)

    for item in SAMPLES:
        out_file = os.path.join(OUTPUT_DIR, f"{item['id']}.mp3")
        communicate = edge_tts.Communicate(
            text=item["text"],
            voice=item["voice"],
            rate=item["rate"],
            pitch=item["pitch"]
        )
        await communicate.save(out_file)
        size_kb = os.path.getsize(out_file) / 1024
        print(f"✅ [{item['title']}] -> {item['id']}.mp3 ({size_kb:.1f} KB)")

    print("\n" + "=" * 70)
    print("🎉 全部小帅解说样音生成完成！")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(generate_all())
