"""Scenario generation via LLM with robust fallback and story synthesis.

Also supports story-first flow:
- generate_story: produce ~300-word story
- panels_from_story: derive panel JSON from story text
"""
from __future__ import annotations
import json, re, datetime
from typing import Any, List, Dict
from openai import OpenAI


def safe_parse_json(raw: str):
    raw = re.sub(r",\s*([\]}])", r"\1", raw)
    raw = raw.replace("\n", " ").replace("\r", " ")
    try:
        return json.loads(raw)
    except Exception:
        return []


def _fallback_scenario(config) -> List[Dict[str, Any]]:
    n = max(2, int(getattr(config.scenario, "panels", 6)))
    ctx = getattr(config.scenario, "setting", "city at night")
    prot = getattr(config.scenario, "protagonist", "Hero")
    ant = getattr(config.scenario, "antagonist", "Nemesis")
    tone = getattr(config.scenario, "tone", "dramatic")
    panels: List[Dict[str, Any]] = []
    beats = [
        (prot, f"{prot} moves through {ctx}, eyes scanning."),
        (ant, f"{ant} watches from the shadows."),
        (prot, "Did you think I'd miss the trail?"),
        (ant, "You're already too late."),
        (None, "Neon rain cuts across the alley; tension spikes."),
        (prot, "Then I'll change the clock."),
        (ant, "Try me."),
        (None, "A glass pane cracks; the city holds its breath."),
    ]
    for i in range(n):
        who, line = beats[i % len(beats)]
        panels.append({
            "context": ctx,
            "scene": [tone, "neon", "rain", "alley", "tension"],
            "speaker": who or "Narrator",
            "speech": line,
        })
    return panels


def _fallback_story(config, target_words: int = 300) -> str:
    prot = getattr(config.scenario, "protagonist", "Hero")
    ant = getattr(config.scenario, "antagonist", "Nemesis")
    genre = getattr(config.scenario, "genre", "action")
    setting = getattr(config.scenario, "setting", "city at night")
    tone = getattr(config.scenario, "tone", "dramatic")
    title = getattr(config.scenario, "episode_title", "Episode")
    base = (
        f"{title}. In the {setting}, {prot} moves with purpose while rumors of {ant} spread. "
        f"Under a {tone} mood, streets hum with the {genre} pulse. {prot} discovers a clue—subtle as neon reflected in rain—"
        f"leading closer to {ant}. A confrontation simmers in alleys and rooftops; promises made, debts recalled. "
        f"As the night deepens, choices narrow. In the final moment, {ant} reveals a twist that forces {prot} to "
        f"reconsider everything: allies, motives, and the cost of truth."
    )
    return base


def generate_story(config, client: OpenAI | None, target_words: int = 300) -> str:
    """Generate ~target_words words of story prose using LLM or fallback."""
    if client is None:
        return _fallback_story(config, target_words)
    completion = client.chat.completions.create(
        model=config.model.llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a seasoned webtoon writer. Write a concise, vivid prose STORY around {words} words. "
                    "Do NOT include panels or lists; produce prose paragraphs only.".replace("{words}", str(target_words))
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Title: {getattr(config.scenario, 'episode_title', 'Episode')}\n"
                    f"Genre: {getattr(config.scenario, 'genre', 'action')}\n"
                    f"Setting: {getattr(config.scenario, 'setting', 'city at night')}\n"
                    f"Tone: {getattr(config.scenario, 'tone', 'dramatic')}\n"
                    f"Protagonist: {getattr(config.scenario, 'protagonist', 'Hero')}\n"
                    f"Antagonist: {getattr(config.scenario, 'antagonist', 'Nemesis')}\n"
                    "Please write a self-contained story with a beginning, rising tension, and a hook at the end."
                ),
            },
        ],
        temperature=getattr(config.generation, "temperature", 0.8),
        max_tokens=max(512, getattr(config.generation, "max_tokens", 500)),
    )
    return completion.choices[0].message.content.strip()


def panels_from_story(config, client: OpenAI | None, story_text: str) -> List[Dict[str, Any]]:
    """Convert story prose into a JSON list of panels.

    Returns exactly config.scenario.panels panels using LLM or a fallback splitter.
    """
    n = max(2, int(getattr(config.scenario, "panels", 6)))
    if client is None:
        # Fallback: naive split by sentences into n groups
        import re
        sentences = re.split(r"(?<=[.!?])\s+", story_text.strip()) if story_text else []
        if not sentences:
            return _fallback_scenario(config)
        chunk_size = max(1, len(sentences) // n)
        chunks = [" ".join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]
        chunks = (chunks + [""] * n)[:n]
        prot = getattr(config.scenario, "protagonist", "Hero")
        ant = getattr(config.scenario, "antagonist", "Nemesis")
        panels: List[Dict[str, Any]] = []
        for ch in chunks:
            who = "Narrator"
            ch_low = ch.lower()
            if prot.lower() in ch_low:
                who = prot
            elif ant.lower() in ch_low:
                who = ant
            tag_candidates = ["night","neon","rain","alley","city","rooftop","shadow","chase","fight","quiet"]
            tags = [t for t in tag_candidates if t in ch_low][:5] or ["scene"]
            panels.append({
                "context": getattr(config.scenario, "setting", "city"),
                "scene": tags,
                "speaker": who,
                "speech": ch[:120].strip() or "...",
            })
        return panels

    # LLM path
    completion = client.chat.completions.create(
        model=config.model.llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a webtoon script adapter. Given a STORY, return ONLY a JSON list with exactly {n} panels. "
                    "Each panel is an object with keys: context (string), scene (array of 3-6 short visual tags), "
                    "speaker (string), speech (short dialogue or narration). Keep speech under 25 words.".replace("{n}", str(n))
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Story:\n{story_text}\n\n"
                    f"Setting: {getattr(config.scenario, 'setting', 'city at night')} | "
                    f"Tone: {getattr(config.scenario, 'tone', 'dramatic')} | "
                    f"Protagonist: {getattr(config.scenario, 'protagonist', 'Hero')} | "
                    f"Antagonist: {getattr(config.scenario, 'antagonist', 'Nemesis')}\n"
                    f"Return exactly {n} panels as JSON list."
                ),
            },
        ],
        temperature=getattr(config.generation, "temperature", 0.8),
        max_tokens=max(768, getattr(config.generation, "max_tokens", 500)),
    )
    raw = completion.choices[0].message.content.strip()
    scenes = safe_parse_json(raw)
    if isinstance(scenes, list) and len(scenes) == n:
        return scenes
    return _fallback_scenario(config)


def generate_scenario(config, client: OpenAI | None):
    if client is None:
        return _fallback_scenario(config)
    completion = client.chat.completions.create(
        model=config.model.llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a seasoned webtoon scriptwriter. "
                    "Return ONLY a JSON list of panels for a Korean manhwa episode. "
                    "Each panel is a dictionary with keys: context, scene, speaker, speech. "
                    "The 'scene' must be an array of short visual tags. "
                    "Write exactly {panels} panels with a coherent arc. "
                    "Genre: {genre}; Setting: {setting}; Tone: {tone}; Protagonist: {protagonist}; Antagonist: {antagonist}. "
                    "Example: [ {\"context\":\"Village\",\"scene\":[\"mountain\"],\"speaker\":\"Akira\",\"speech\":\"...\"} ]"
                )
            },
            {"role": "user", "content": (
                f"Title: {config.scenario.episode_title}\n"
                f"Please write {config.scenario.panels} panels that build tension and end with a hook."
            )},
        ],
        temperature=getattr(config.generation, "temperature", 0.8),
        max_tokens=getattr(config.generation, "max_tokens", 500),
    )
    raw = completion.choices[0].message.content.strip()
    scenes = safe_parse_json(raw)
    if isinstance(scenes, dict) and "panels" in scenes:
        panels = scenes["panels"]
        if not isinstance(panels, list) or len(panels) == 0:
            return _fallback_scenario(config)
        return panels
    elif isinstance(scenes, list):
        return scenes or _fallback_scenario(config)
    else:
        return _fallback_scenario(config)


def synthesize_prose_story(config, scenes) -> str:
    title = getattr(config.scenario, "episode_title", "Episode")
    lines = [title, ""]
    for i, sc in enumerate(scenes, 1):
        ctx = sc.get("context", "")
        scene = sc.get("scene", [])
        if isinstance(scene, list):
            scene = ", ".join(scene)
        speaker = sc.get("speaker", "")
        speech = sc.get("speech", "")
        lines.append(f"Panel {i}: [{ctx}] ({scene})")
        lines.append(f"{speaker}: {speech}" if speaker else f"{speech}")
        lines.append("")
    return "\n".join(lines)
