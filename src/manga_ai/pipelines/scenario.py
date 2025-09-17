"""Scenario generation via LLM with robust fallback and story synthesis.

Also supports story-first flow:
- generate_story: produce ~300-word story
- panels_from_story: derive panel JSON from story text
"""
from __future__ import annotations
import json, re, datetime
from typing import Any, List, Dict
import os
import random
import random, os
from openai import OpenAI


def safe_parse_json(raw: str):
    raw = re.sub(r",\s*([\]}])", r"\1", raw)
    raw = raw.replace("\n", " ").replace("\r", " ")
    try:
        return json.loads(raw)
    except Exception:
        return []


def _maybe_seed(config):
    seed = getattr(config.scenario, "random_seed", None)
    if seed is not None:
        random.seed(seed)


def _rand_from(options: List[str]) -> str:
    return random.choice(options)


def _random_defaults(config):
    """Optionally override scenario defaults with random elements if enabled."""
    if not getattr(config.scenario, "randomize", True):
        return
    _maybe_seed(config)
    settings = [
        "ancient temple in misty mountains",
        "cyberpunk alley under neon rain",
        "quiet seaside town at dawn",
        "desert bazaar at high noon",
        "snowy rooftop over a sleeping city",
        "forest shrine at twilight",
        "space station maintenance bay",
        "subway platform during rush hour",
    ]
    genres = ["action","mystery","drama","romance","horror","fantasy","sci-fi","thriller"]
    tones = ["dramatic","somber","hopeful","tense","whimsical","melancholic","grim","uplifting"]
    names = ["Jin","Mina","Akira","Yuri","Hana","Kai","Rin","Kira","June","Sora","Raven","Vex","Aria"]
    ep_prefix = ["Episode","Chapter","Act","Stage"]
    if not os.getenv("SETTING"):
        config.scenario.setting = _rand_from(settings)
    if not os.getenv("GENRE"):
        config.scenario.genre = _rand_from(genres)
    if not os.getenv("TONE"):
        config.scenario.tone = _rand_from(tones)
    if not os.getenv("PROTAGONIST"):
        config.scenario.protagonist = _rand_from(names)
    if not os.getenv("ANTAGONIST"):
        config.scenario.antagonist = _rand_from([n for n in names if n != config.scenario.protagonist])
    if not os.getenv("EPISODE_TITLE"):
        config.scenario.episode_title = f"{_rand_from(ep_prefix)} {random.randint(1, 999)}: {_rand_from(['Rooftop Oath','Broken Code','Silent Tide','Neon Promise','Fallen Signal','Ashen Vow'])}"


def _fallback_scenario(config) -> List[Dict[str, Any]]:
    _random_defaults(config)
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
    _random_defaults(config)
    prot = getattr(config.scenario, "protagonist", "Hero")
    ant = getattr(config.scenario, "antagonist", "Nemesis")
    genre = getattr(config.scenario, "genre", "action")
    setting = getattr(config.scenario, "setting", "city at night")
    tone = getattr(config.scenario, "tone", "dramatic")
    title = getattr(config.scenario, "episode_title", "Episode")
    # Compose a concise base and expand with flavor until reaching target_words
    parts = [
        f"{title}. In the {setting}, {prot} moves with purpose while rumors of {ant} spread.",
        f"Under a {tone} mood, streets hum with a {genre} pulse.",
        f"{prot} discovers a clue—subtle as neon reflected in rain—leading closer to {ant}.",
        f"Promises linger; debts resurface. A confrontation simmers in alleys and rooftops.",
        f"As the night deepens, choices narrow; each step echoes against glass and concrete.",
        f"Finally, {ant} reveals a twist that forces {prot} to reconsider allies, motives, and the cost of truth."
    ]
    flavor = [
        "Footsteps blur with sirens; the skyline watches, indifferent yet witness to every vow.",
        f"Old messages flicker on cracked screens; {prot} reads between the flickers for intent.",
        f"Street vendors pack in a hurry, trading rumors of {ant} for folded bills and wary glances.",
        "Rain scatters light into sharp shards; the city becomes a prism of suspicion and resolve.",
        "Whispers flow through stairwells, mapping a maze that returns always to the same door.",
        f"In a brief stillness, {prot} remembers why this began and what it must cost to finish.",
        f"Every clue ties to another, a knot tightening around {ant}'s name and the night itself.",
        f"From the rooftop edge, {prot} counts breaths, then leaps—not down, but toward an answer."
    ]
    text = " ".join(parts)
    def _wc(s: str) -> int:
        return len((s or "").split())
    i = 0
    target = max(50, int(target_words) if target_words else 300)
    while _wc(text) < target and i < 60:
        text += " " + random.choice(flavor)
        i += 1
    return text


def generate_story(config, client: OpenAI | None, target_words: int = 300) -> str:
    """Generate ~target_words words of story prose using LLM or fallback, guaranteeing minimum length."""
    # Apply optional randomization of scenario parameters before generation
    _random_defaults(config)

    def _count_words(s: str) -> int:
        return len((s or "").split())

    # If no client, use robust fallback that expands to target length
    if client is None:
        return _fallback_story(config, target_words)

    # Scale token budget to accommodate the requested word count
    cfg_max = int(getattr(config.generation, "max_tokens", 500) or 500)
    est_tokens = int(max(512, (target_words or 300) * 2.0))
    max_toks = max(cfg_max, est_tokens)
    max_toks = min(max_toks, 2048)

    def _ask_story(words: int) -> str:
        completion = client.chat.completions.create(
            model=config.model.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a seasoned webtoon writer. Write a vivid prose STORY of at least {words} words. "
                        "Do NOT include panels or lists; output prose paragraphs only."
                    ).replace("{words}", str(max(50, int(words or 300))))
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
                        "Write a self-contained story with a beginning, rising tension, and a hook at the end."
                    ),
                },
            ],
            temperature=getattr(config.generation, "temperature", 0.8),
            max_tokens=max_toks,
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
