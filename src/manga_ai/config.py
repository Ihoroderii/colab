from dataclasses import dataclass
from typing import List
import os


@dataclass
class ModelConfig:
    stable_diffusion_model: str = "stabilityai/stable-diffusion-3.5-large"
    llm_model: str = "meta-llama/Meta-Llama-3-8B-Instruct:novita"
    blip_model: str = "Salesforce/blip-image-captioning-large"
    clip_model: str = "ViT-B/32"
    device: str = "cuda"
    torch_dtype: str = "float16"
    safety_checker: bool = False
    REMOVED_TOKENtoken: str | None = None


@dataclass
class GenerationConfig:
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    max_sequence_length: int = 77
    temperature: float = 0.8
    max_tokens: int = 500


@dataclass
class StyleConfig:
    panel_width: int = 800
    panel_gap: int = 40
    panel_width_jitter: float = 0.0  # 0.0..0.95 random +/- variation of width
    auto_panel_width: bool = True  # if True, compute width from scene/speech heuristics
    font_size: int = 24
    bold_font_size: int = 20
    font_paths: List[str] | None = None
    canvas_bg: str = "#ffffff"  # background for assembled chapter
    # Optional lighting effect controls
    apply_dramatic_lighting: bool = False
    lighting_style: str = "dramatic"  # 'dramatic' | 'soft' | 'contrast'
    lighting_intensity: float = 0.5
    # Auto style (derive multiple style params from scenario)
    auto_style: bool = True
    # Detection-assisted placement
    auto_bubble_placement: bool = True
    draw_detection_debug: bool = False
    # Panel border
    panel_border_width: int = 4  # 0 disables border
    panel_border_color: str = "#000000"
    # Square panels
    square_panels: bool = True
    square_size: int | None = None  # If None, use computed target_width per panel
    square_fill_color: str = "#ffffff"
    # Keep all panels the same size in a run
    same_panel_size: bool = True
    # Runtime-computed fixed side (set in runner), not from env
    runtime_square_size: int | None = None

    def __post_init__(self):
        if self.font_paths is None:
            self.font_paths = [
                "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                "/usr/local/share/fonts/NanumGothic.ttf",
                "./fonts/NanumGothic.ttf",
            ]


@dataclass
class OutputConfig:
    output_dir: str = "output"
    save_individual_panels: bool = True
    panel_filename_template: str = "panel_{}.png"
    chapter_filename: str = "manhwa_chapter.png"
    image_quality: int = 95


@dataclass
class ScenarioConfig:
    panels: int = 4
    genre: str = "action"
    setting: str = "modern city at dusk"
    episode_title: str = "Episode 1: Rooftop Oath"
    tone: str = "dramatic"
    protagonist: str = "Jin"
    antagonist: str = "Raven"
    # Randomization controls for fallback (no-LLM) generation
    randomize: bool = True
    random_seed: int | None = None


@dataclass
class ExportConfig:
    enabled: bool = True
    width: int = 800
    max_slice_height: int = 1280
    overlap: int = 40
    format: str = "png"
    quality: int = 95
    output_subdir: str = "export"


@dataclass
class ValidationConfig:
    enabled: bool = False
    similarity_threshold: float = 0.7
    blip_model: str = "Salesforce/blip-image-captioning-large"
    clip_model: str = "ViT-B/32"


class Config:
    def __init__(self):
        self.model = ModelConfig()
        self.generation = GenerationConfig()
        self.style = StyleConfig()
        self.output = OutputConfig()
        self.scenario = ScenarioConfig()
        self.export = ExportConfig()
        self.validation = ValidationConfig()
        os.makedirs(self.output.output_dir, exist_ok=True)

    @classmethod
    def from_env(cls) -> "Config":
        c = cls()
        # Model
        sd_model = os.getenv("SD_MODEL")
        print("sd_model : ", sd_model)
        if sd_model:
            c.model.stable_diffusion_model = sd_model
        device = os.getenv("DEVICE")
        if device:
            c.model.device = device
        llm_model = os.getenv("LLM_MODEL")
        if llm_model:
            c.model.llm_model = llm_model
        c.model.REMOVED_TOKENtoken = (
            os.getenv("HF_TOKEN")
            or os.getenv("HF_API_KEY")
            or os.getenv("HUGGING_FACE_HUB_TOKEN")
            or os.getenv("OPENAI_API_KEY")
        )

        # Generation
        guidance = os.getenv("GUIDANCE_SCALE")
        if guidance:
            c.generation.guidance_scale = float(guidance)
        num_steps = os.getenv("NUM_STEPS")
        if num_steps:
            c.generation.num_inference_steps = int(num_steps)

        # Style
        panel_w = os.getenv("PANEL_WIDTH")
        if panel_w:
            c.style.panel_width = int(panel_w)
        panel_gap = os.getenv("PANEL_GAP")
        if panel_gap:
            c.style.panel_gap = int(panel_gap)
        pw_jitter = os.getenv("PANEL_WIDTH_JITTER")
        if pw_jitter:
            c.style.panel_width_jitter = float(pw_jitter)
        apw = os.getenv("AUTO_PANEL_WIDTH")
        if apw:
            c.style.auto_panel_width = apw.lower() in ("1", "true", "yes", "on")
        apply_light = os.getenv("APPLY_DRAMATIC_LIGHTING")
        if apply_light:
            c.style.apply_dramatic_lighting = apply_light.lower() in ("1","true","yes","on")
        light_style = os.getenv("LIGHTING_STYLE")
        if light_style:
            c.style.lighting_style = light_style
        light_int = os.getenv("LIGHTING_INTENSITY")
        if light_int:
            c.style.lighting_intensity = float(light_int)
        auto_style = os.getenv("AUTO_STYLE") or os.getenv("AUTO_TUNE_STYLE")
        if auto_style:
            c.style.auto_style = auto_style.lower() in ("1","true","yes","on")
        abp = os.getenv("AUTO_BUBBLE_PLACEMENT")
        if abp:
            c.style.auto_bubble_placement = abp.lower() in ("1","true","yes","on")
        dbg = os.getenv("DRAW_DETECTION_DEBUG")
        if dbg:
            c.style.draw_detection_debug = dbg.lower() in ("1","true","yes","on")
        canvas_bg = os.getenv("CANVAS_BG")
        if canvas_bg:
            c.style.canvas_bg = canvas_bg
        panel_border_w = os.getenv("PANEL_BORDER_WIDTH")
        if panel_border_w:
            c.style.panel_border_width = int(panel_border_w)
        panel_border_color = os.getenv("PANEL_BORDER_COLOR")
        if panel_border_color:
            c.style.panel_border_color = panel_border_color
        square = os.getenv("SQUARE_PANELS")
        if square:
            c.style.square_panels = square.lower() in ("1","true","yes","on")
        square_size = os.getenv("SQUARE_SIZE")
        if square_size:
            c.style.square_size = int(square_size)
        square_fill = os.getenv("SQUARE_FILL_COLOR")
        if square_fill:
            c.style.square_fill_color = square_fill
        same_size = os.getenv("SAME_PANEL_SIZE")
        if same_size:
            c.style.same_panel_size = same_size.lower() in ("1","true","yes","on")

        # Output
        out_dir = os.getenv("OUTPUT_DIR")
        if out_dir:
            c.output.output_dir = out_dir
            os.makedirs(c.output.output_dir, exist_ok=True)

        # Scenario
        panels = os.getenv("PANELS")
        if panels:
            c.scenario.panels = int(panels)
        genre = os.getenv("GENRE")
        if genre:
            c.scenario.genre = genre
        setting = os.getenv("SETTING")
        if setting:
            c.scenario.setting = setting
        ep = os.getenv("EPISODE_TITLE")
        if ep:
            c.scenario.episode_title = ep
        tone = os.getenv("TONE")
        if tone:
            c.scenario.tone = tone
        prot = os.getenv("PROTAGONIST")
        if prot:
            c.scenario.protagonist = prot
        ant = os.getenv("ANTAGONIST")
        if ant:
            c.scenario.antagonist = ant
        rnd = os.getenv("RANDOMIZE")
        if rnd:
            c.scenario.randomize = rnd.lower() in ("1","true","yes","on")
        seed = os.getenv("SEED") or os.getenv("RANDOM_SEED")
        if seed:
            try:
                c.scenario.random_seed = int(seed)
            except Exception:
                pass

        # Export
        export_enabled = os.getenv("EXPORT_ENABLED")
        if export_enabled:
            c.export.enabled = export_enabled.lower() in ("1", "true", "yes")
        export_w = os.getenv("EXPORT_WIDTH")
        if export_w:
            c.export.width = int(export_w)
        export_h = os.getenv("EXPORT_MAX_SLICE_HEIGHT")
        if export_h:
            c.export.max_slice_height = int(export_h)
        export_overlap = os.getenv("EXPORT_OVERLAP")
        if export_overlap:
            c.export.overlap = int(export_overlap)
        export_fmt = os.getenv("EXPORT_FORMAT")
        if export_fmt:
            c.export.format = export_fmt
        export_quality = os.getenv("EXPORT_QUALITY")
        if export_quality:
            c.export.quality = int(export_quality)
        export_subdir = os.getenv("EXPORT_SUBDIR")
        if export_subdir:
            c.export.output_subdir = export_subdir

        # Validation
        val_enabled = os.getenv("VALIDATION_ENABLED")
        if val_enabled:
            c.validation.enabled = val_enabled.lower() in ("1", "true", "yes")
        val_thr = os.getenv("VALIDATION_THRESHOLD")
        if val_thr:
            c.validation.similarity_threshold = float(val_thr)
        blip = os.getenv("BLIP_MODEL")
        if blip:
            c.validation.blip_model = blip
        clip = os.getenv("CLIP_MODEL")
        if clip:
            c.validation.clip_model = clip

        return c
