from dataclasses import dataclass
from typing import List
import os


def _load_env_file(path: str = ".env") -> None:
    """Load simple KEY=VALUE lines without overriding existing environment."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


@dataclass
class ModelConfig:
    image_backend: str = "local"  # 'local' uses diffusers; 'api' uses a remote image API
    llm_provider: str = "huggingface"  # 'huggingface', 'cloudflare', or 'none'
    stable_diffusion_model: str = "stabilityai/stable-diffusion-3.5-large"
    fallback_diffusion_model: str = "runwayml/stable-diffusion-v1-5"
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


@dataclass
class StyleNormalizationConfig:
    enabled: bool = False
    reference_path: str = "references/manga_style_reference.png"
    output_subdir: str = "normalized"
    style_id: str = "manga_bw_v1"
    width: int = 768
    height: int = 1024
    gamma: float = 0.96
    contrast: float = 1.08
    autocontrast_cutoff: float = 1.0
    sharpen_radius: float = 1.2
    sharpen_percent: int = 110
    sharpen_threshold: int = 3
    grain_strength: float = 0.02
    screentone_style: str = "dots_medium"
    page_autocontrast_cutoff: float = 0.3
    page_contrast: float = 1.02


@dataclass
class BubbleConfig:
    backend: str = "manhwa_bubbles"  # 'internal' or 'manhwa_bubbles'
    project_path: str = "../bubble"
    use_yolo: bool = False
    prefer_cairo: bool = True
    fallback_to_internal: bool = True


@dataclass
class ReferenceConfig:
    image_path: str | None = None
    img2img_strength: float = 0.38
    resize_mode: str = "fit"  # 'fit' preserves the whole reference; 'crop' fills the panel


@dataclass
class BlenderControlConfig:
    enabled: bool = False
    executable: str = "blender"
    output_subdir: str = "control"
    render_width: int = 768
    render_height: int = 1024
    img2img_strength: float = 0.34
    fallback_to_pil: bool = True


@dataclass
class ImageApiConfig:
    provider: str = "openai"
    model: str = "gpt-image-1"
    api_key: str | None = None
    base_url: str | None = None
    hf_provider: str | None = None
    cloudflare_account_id: str | None = None
    cloudflare_api_token: str | None = None
    cloudflare_use_multipart: bool = True
    size: str = "1024x1024"
    quality: str = "medium"
    output_format: str = "png"


class Config:
    def __init__(self):
        self.model = ModelConfig()
        self.generation = GenerationConfig()
        self.style = StyleConfig()
        self.output = OutputConfig()
        self.scenario = ScenarioConfig()
        self.export = ExportConfig()
        self.validation = ValidationConfig()
        self.style_normalization = StyleNormalizationConfig()
        self.bubbles = BubbleConfig()
        self.reference = ReferenceConfig()
        self.blender = BlenderControlConfig()
        self.image_api = ImageApiConfig()
        os.makedirs(self.output.output_dir, exist_ok=True)

    @classmethod
    def from_env(cls) -> "Config":
        _load_env_file()
        c = cls()
        # Model
        sd_model = os.getenv("SD_MODEL")
        if sd_model:
            c.model.stable_diffusion_model = sd_model
        image_backend = os.getenv("IMAGE_BACKEND")
        if image_backend:
            c.model.image_backend = image_backend.lower()
        fallback_sd_model = os.getenv("FALLBACK_SD_MODEL")
        if fallback_sd_model:
            c.model.fallback_diffusion_model = fallback_sd_model
        device = os.getenv("DEVICE")
        if device:
            c.model.device = device
        llm_model = os.getenv("LLM_MODEL")
        if llm_model:
            c.model.llm_model = llm_model
        llm_provider = os.getenv("LLM_PROVIDER")
        if llm_provider:
            c.model.llm_provider = llm_provider.lower()
        c.model.REMOVED_TOKENtoken = (
            os.getenv("HF_TOKEN")
            or os.getenv("HF_API_KEY")
            or os.getenv("HUGGING_FACE_HUB_TOKEN")
            or os.getenv("LLM_API_KEY")
            or (os.getenv("OPENAI_API_KEY") if c.model.image_backend != "api" else None)
            or (os.getenv("CLOUDFLARE_API_TOKEN") if c.model.llm_provider == "cloudflare" else None)
        )
        image_api_provider = os.getenv("IMAGE_API_PROVIDER")
        if image_api_provider:
            c.image_api.provider = image_api_provider.lower()
        c.image_api.api_key = (
            os.getenv("IMAGE_API_KEY")
            or os.getenv("OPENAI_IMAGE_API_KEY")
            or (os.getenv("OPENAI_API_KEY") if c.image_api.provider == "openai" else None)
            or (os.getenv("HF_TOKEN") if c.image_api.provider == "huggingface" else None)
            or (os.getenv("CLOUDFLARE_API_TOKEN") if c.image_api.provider == "cloudflare" else None)
        )
        image_api_model = os.getenv("IMAGE_API_MODEL")
        if image_api_model:
            c.image_api.model = image_api_model
        image_api_base_url = os.getenv("IMAGE_API_BASE_URL")
        if image_api_base_url:
            c.image_api.base_url = image_api_base_url
        hf_image_provider = os.getenv("HF_IMAGE_PROVIDER") or os.getenv("HUGGINGFACE_IMAGE_PROVIDER")
        if hf_image_provider:
            c.image_api.hf_provider = hf_image_provider
        cloudflare_account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        if cloudflare_account_id:
            c.image_api.cloudflare_account_id = cloudflare_account_id
        cloudflare_api_token = os.getenv("CLOUDFLARE_API_TOKEN")
        if cloudflare_api_token:
            c.image_api.cloudflare_api_token = cloudflare_api_token
        cloudflare_multipart = os.getenv("CLOUDFLARE_USE_MULTIPART")
        if cloudflare_multipart:
            c.image_api.cloudflare_use_multipart = cloudflare_multipart.lower() in ("1", "true", "yes", "on")
        image_api_size = os.getenv("IMAGE_API_SIZE")
        if image_api_size:
            c.image_api.size = image_api_size
        image_api_quality = os.getenv("IMAGE_API_QUALITY")
        if image_api_quality:
            c.image_api.quality = image_api_quality
        image_api_output_format = os.getenv("IMAGE_API_OUTPUT_FORMAT")
        if image_api_output_format:
            c.image_api.output_format = image_api_output_format.lower()

        # Generation
        guidance = os.getenv("GUIDANCE_SCALE")
        if guidance:
            c.generation.guidance_scale = float(guidance)
        num_steps = os.getenv("NUM_STEPS") or os.getenv("NUM_INFERENCE_STEPS")
        if num_steps:
            c.generation.num_inference_steps = int(num_steps)
        max_sequence_length = os.getenv("MAX_SEQUENCE_LENGTH")
        if max_sequence_length:
            c.generation.max_sequence_length = int(max_sequence_length)
        temperature = os.getenv("TEMPERATURE")
        if temperature:
            c.generation.temperature = float(temperature)
        max_tokens = os.getenv("MAX_TOKENS")
        if max_tokens:
            c.generation.max_tokens = int(max_tokens)

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
        font_size = os.getenv("FONT_SIZE")
        if font_size:
            c.style.font_size = int(font_size)
        bold_font_size = os.getenv("BOLD_FONT_SIZE")
        if bold_font_size:
            c.style.bold_font_size = int(bold_font_size)
        font_path = os.getenv("FONT_PATH")
        if font_path:
            c.style.font_paths = [font_path]

        # Output
        out_dir = os.getenv("OUTPUT_DIR")
        if out_dir:
            c.output.output_dir = out_dir
            os.makedirs(c.output.output_dir, exist_ok=True)
        save_panels = os.getenv("SAVE_INDIVIDUAL_PANELS")
        if save_panels:
            c.output.save_individual_panels = save_panels.lower() in ("1", "true", "yes", "on")
        panel_filename_template = os.getenv("PANEL_FILENAME_TEMPLATE")
        if panel_filename_template:
            c.output.panel_filename_template = panel_filename_template
        chapter_filename = os.getenv("CHAPTER_FILENAME")
        if chapter_filename:
            c.output.chapter_filename = chapter_filename
        image_quality = os.getenv("IMAGE_QUALITY")
        if image_quality:
            c.output.image_quality = int(image_quality)

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

        # Reference image conditioning
        ref_img = os.getenv("REFERENCE_IMAGE") or os.getenv("REFERENCE_IMAGE_PATH")
        if ref_img:
            c.reference.image_path = ref_img
        strength = os.getenv("IMG2IMG_STRENGTH") or os.getenv("REFERENCE_STRENGTH")
        if strength:
            c.reference.img2img_strength = max(0.0, min(1.0, float(strength)))
        resize_mode = os.getenv("REFERENCE_RESIZE_MODE")
        if resize_mode:
            c.reference.resize_mode = resize_mode.lower()

        # Blender scene/control-map generation
        blender_enabled = os.getenv("USE_BLENDER_CONTROL") or os.getenv("BLENDER_CONTROL_ENABLED")
        if blender_enabled:
            c.blender.enabled = blender_enabled.lower() in ("1", "true", "yes", "on")
        blender_executable = os.getenv("BLENDER_EXECUTABLE")
        if blender_executable:
            c.blender.executable = blender_executable
        blender_output_subdir = os.getenv("BLENDER_CONTROL_SUBDIR")
        if blender_output_subdir:
            c.blender.output_subdir = blender_output_subdir
        blender_width = os.getenv("BLENDER_CONTROL_WIDTH")
        if blender_width:
            c.blender.render_width = int(blender_width)
        blender_height = os.getenv("BLENDER_CONTROL_HEIGHT")
        if blender_height:
            c.blender.render_height = int(blender_height)
        blender_strength = os.getenv("BLENDER_CONTROL_STRENGTH")
        if blender_strength:
            c.blender.img2img_strength = max(0.0, min(1.0, float(blender_strength)))
        blender_fallback = os.getenv("BLENDER_CONTROL_FALLBACK")
        if blender_fallback:
            c.blender.fallback_to_pil = blender_fallback.lower() in ("1", "true", "yes", "on")

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

        # Style normalization
        style_norm_enabled = os.getenv("STYLE_NORMALIZATION_ENABLED") or os.getenv("NORMALIZE_MANGA_STYLE")
        if style_norm_enabled:
            c.style_normalization.enabled = style_norm_enabled.lower() in ("1", "true", "yes", "on")
        style_ref = os.getenv("STYLE_REFERENCE_IMAGE") or os.getenv("MANGA_STYLE_REFERENCE")
        if style_ref:
            c.style_normalization.reference_path = style_ref
        style_norm_subdir = os.getenv("STYLE_NORMALIZATION_SUBDIR")
        if style_norm_subdir:
            c.style_normalization.output_subdir = style_norm_subdir
        style_norm_width = os.getenv("STYLE_NORMALIZATION_WIDTH")
        if style_norm_width:
            c.style_normalization.width = int(style_norm_width)
        style_norm_height = os.getenv("STYLE_NORMALIZATION_HEIGHT")
        if style_norm_height:
            c.style_normalization.height = int(style_norm_height)
        style_norm_gamma = os.getenv("STYLE_NORMALIZATION_GAMMA")
        if style_norm_gamma:
            c.style_normalization.gamma = float(style_norm_gamma)
        style_norm_contrast = os.getenv("STYLE_NORMALIZATION_CONTRAST")
        if style_norm_contrast:
            c.style_normalization.contrast = float(style_norm_contrast)
        style_norm_cutoff = os.getenv("STYLE_NORMALIZATION_AUTOCONTRAST_CUTOFF")
        if style_norm_cutoff:
            c.style_normalization.autocontrast_cutoff = float(style_norm_cutoff)
        style_norm_grain = os.getenv("STYLE_NORMALIZATION_GRAIN_STRENGTH")
        if style_norm_grain:
            c.style_normalization.grain_strength = float(style_norm_grain)

        # Bubble rendering
        bubble_backend = os.getenv("BUBBLE_BACKEND")
        if bubble_backend:
            c.bubbles.backend = bubble_backend.lower()
        bubble_project_path = os.getenv("BUBBLE_PROJECT_PATH")
        if bubble_project_path:
            c.bubbles.project_path = bubble_project_path
        bubble_use_yolo = os.getenv("BUBBLE_USE_YOLO")
        if bubble_use_yolo:
            c.bubbles.use_yolo = bubble_use_yolo.lower() in ("1", "true", "yes", "on")
        bubble_prefer_cairo = os.getenv("BUBBLE_PREFER_CAIRO")
        if bubble_prefer_cairo:
            c.bubbles.prefer_cairo = bubble_prefer_cairo.lower() in ("1", "true", "yes", "on")
        bubble_fallback = os.getenv("BUBBLE_FALLBACK_TO_INTERNAL")
        if bubble_fallback:
            c.bubbles.fallback_to_internal = bubble_fallback.lower() in ("1", "true", "yes", "on")

        return c
