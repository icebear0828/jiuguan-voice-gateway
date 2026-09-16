#!/usr/bin/env python3
"""
GPT-SoVITS Model Version Manager & CLI Synthesizer
Provides unified lifecycle management for multiple anime/game voice models:
- Version listing & switching
- Model registration & packaging
- Config inspection
- Direct multi-version batch inference
"""

import os
import sys
import json
import argparse
import shutil
from typing import Dict, Any, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "assets/models")
REGISTRY_PATH = os.path.join(MODELS_DIR, "registry.json")
SOVITS_ROOT = os.environ.get("SOVITS_ROOT", os.path.abspath(os.path.join(PROJECT_ROOT, "..", "GPT-SoVITS")))
SOVITS_CORE = os.path.join(SOVITS_ROOT, "GPT_SoVITS")


def load_registry() -> Dict[str, Any]:
    if not os.path.exists(REGISTRY_PATH):
        return {"schema_version": "1.0", "default_character": "", "characters": {}}
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(data: Dict[str, Any]):
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_model_info(character_id: str, version: Optional[str] = None) -> Dict[str, Any]:
    registry = load_registry()
    chars = registry.get("characters", {})
    if character_id not in chars:
        raise ValueError(f"Character '{character_id}' not found in registry. Available: {list(chars.keys())}")
    
    char_entry = chars[character_id]
    target_version = version or char_entry.get("active_version")
    if not target_version or target_version not in char_entry.get("versions", {}):
        raise ValueError(f"Version '{target_version}' not found for character '{character_id}'")
    
    rel_model_dir = char_entry["versions"][target_version]["model_dir"]
    full_model_dir = os.path.join(MODELS_DIR, rel_model_dir)
    info_path = os.path.join(full_model_dir, "model_info.json")
    
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"model_info.json not found at {info_path}")
        
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    
    info["_full_dir"] = full_model_dir
    info["_version_key"] = target_version
    return info


def list_models():
    registry = load_registry()
    print("\n" + "=" * 76)
    print(" 🎙️  GPT-SoVITS 模型版本注册清单 (Model Registry)")
    print("=" * 76)
    default_char = registry.get("default_character", "None")
    print(f"当前系统默认角色: [{default_char}]\n")

    for char_id, char_info in registry.get("characters", {}).items():
        active_ver = char_info.get("active_version", "N/A")
        print(f"📌 角色标识: {char_id}")
        print(f"   角色全称: {char_info.get('character_name')} | CV: {char_info.get('cv', '未知')}")
        print(f"   角色来源: {char_info.get('character_source', '未知')}")
        print(f"   当前激活版本: [ {active_ver} ]")
        print("   可用模型版本列表:")
        for ver_id, ver_data in char_info.get("versions", {}).items():
            is_active = "🌟 (CURRENT)" if ver_id == active_ver else "  "
            status = ver_data.get("status", "unknown")
            desc = ver_data.get("description", "")
            print(f"     {is_active} [{ver_id}] - 状态: {status} | 描述: {desc}")
        print("-" * 76)


def switch_version(character_id: str, version: str):
    registry = load_registry()
    if character_id not in registry.get("characters", {}):
        print(f"❌ 角色 '{character_id}' 不存在！")
        return False
    char_entry = registry["characters"][character_id]
    if version not in char_entry.get("versions", {}):
        print(f"❌ 版本 '{version}' 不存在于角色 '{character_id}' 的注册列表中！")
        return False
    
    char_entry["active_version"] = version
    save_registry(registry)
    
    # Update symlink
    char_dir = os.path.join(MODELS_DIR, character_id)
    current_link = os.path.join(char_dir, "current")
    target_rel = char_entry["versions"][version]["model_dir"].split("/")[-1]
    try:
        if os.path.islink(current_link) or os.path.exists(current_link):
            os.remove(current_link)
        os.symlink(target_rel, current_link)
    except Exception as e:
        print(f"⚠️ 软链接更新警告: {e}")

    print(f"✅ 成功将角色 [{character_id}] 的激活版本切换至: [{version}]")
    return True


def inspect_model(character_id: str, version: Optional[str] = None):
    try:
        info = get_model_info(character_id, version)
    except Exception as e:
        print(f"❌ 获取模型详情失败: {e}")
        return

    print("\n" + "=" * 70)
    print(f" 🔍 模型详细参数报告: {info.get('character_name')} [{info.get('version')}]")
    print("=" * 70)
    print(f"📁 物理存储路径: {info['_full_dir']}")
    print(f"🏗️  底模架构: {info.get('architecture', 'GPT-SoVITS')}")
    print(f"🏷️  当前状态: {info.get('status')}")
    print("\n📦 权重文件:")
    for k, v in info.get("weights", {}).items():
        w_path = os.path.join(info["_full_dir"], v)
        exists = "✅ 存在" if os.path.exists(w_path) else "❌ 缺失"
        print(f"   - {k.upper()}: {v} ({exists})")
    
    print("\n⚙️ 推荐推理参数 (Recommended Inference):")
    for k, v in info.get("recommended_params", {}).items():
        print(f"   - {k}: {v}")
    
    print("\n🎵 预置黄金参考音频 (Prompt Audios):")
    for ref in info.get("reference_audios", []):
        default_tag = " [默认基准]" if ref.get("is_default") else ""
        print(f"   [{ref.get('id')}]{default_tag} {ref.get('name')} ({ref.get('emotion')})")
        print(f"     音频: {ref.get('audio_file')}")
        print(f"     台词: 「{ref.get('text')}」 (语言: {ref.get('lang')})")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-SoVITS Model Version Manager")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # list
    subparsers.add_parser("list", help="List all registered models and versions")

    # switch
    switch_p = subparsers.add_parser("switch", help="Switch active model version for a character")
    switch_p.add_argument("character_id", help="Character identifier (e.g. rimuru, mika)")
    switch_p.add_argument("version", help="Version identifier (e.g. v2pro_master, v1.0)")

    # inspect
    inspect_p = subparsers.add_parser("inspect", help="Inspect model metadata and parameters")
    inspect_p.add_argument("character_id", help="Character identifier")
    inspect_p.add_argument("--version", "-v", default=None, help="Specific version (default: active)")

    args = parser.parse_args()

    if args.command == "list":
        list_models()
    elif args.command == "switch":
        switch_version(args.character_id, args.version)
    elif args.command == "inspect":
        inspect_model(args.character_id, args.version)
    else:
        list_models()
