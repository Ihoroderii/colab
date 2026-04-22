"""CLI for manhwa generator (package version)."""
import argparse
from .config import Config

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manhwa Generator")
    # Model
    parser.add_argument("--sd-model")
    parser.add_argument("--llm-model")
    parser.add_argument("--device", choices=["cuda","cpu"])    
    parser.add_argument("--hf-token")
    # Generation
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--temperature", type=float)
    # Style
    parser.add_argument("--panel-width", type=int)
    parser.add_argument("--panel-gap", type=int)
    parser.add_argument("--panel-width-jitter", type=float, help="0..0.95 random +/- variation around base panel width")
    parser.add_argument("--canvas-bg", help="Canvas background color (e.g., #111111 or 'white')")
    parser.add_argument("--panel-border-width", type=int, help="Border width in pixels around each panel (0 to disable)")
    parser.add_argument("--panel-border-color", help="Border color (e.g., #000000)")
    # Square panels
    sp = parser.add_mutually_exclusive_group()
    sp.add_argument("--square-panels", dest="square_panels", action="store_true", help="Make each panel square by padding")
    sp.add_argument("--no-square-panels", dest="square_panels", action="store_false")
    parser.set_defaults(square_panels=None)
    parser.add_argument("--square-size", type=int, help="Square size in px; if omitted, use computed target width per panel")
    parser.add_argument("--square-fill-color", help="Fill color for square padding (e.g., #ffffff)")
    sp2 = parser.add_mutually_exclusive_group()
    sp2.add_argument("--same-panel-size", dest="same_panel_size", action="store_true", help="Use a single fixed size for all panels in a run")
    sp2.add_argument("--no-same-panel-size", dest="same_panel_size", action="store_false")
    parser.set_defaults(same_panel_size=None)
    apw = parser.add_mutually_exclusive_group()
    apw.add_argument("--auto-panel-width", dest="auto_panel_width", action="store_true", help="Auto compute panel width from scene/speech (default)")
    apw.add_argument("--no-auto-panel-width", dest="auto_panel_width", action="store_false", help="Disable auto width; use --panel-width")
    parser.set_defaults(auto_panel_width=None)
    # Effects
    parser.add_argument("--apply-dramatic-lighting", action="store_true", help="Apply dramatic lighting effect to panels")
    parser.add_argument("--lighting-style", choices=["dramatic","soft","contrast"], help="Lighting style to apply")
    parser.add_argument("--lighting-intensity", type=float, help="Lighting intensity 0..1")
    asg = parser.add_mutually_exclusive_group()
    asg.add_argument("--auto-style", dest="auto_style", action="store_true", help="Auto-derive style (gap/bg/lighting/jitter) from scenario")
    asg.add_argument("--no-auto-style", dest="auto_style", action="store_false", help="Disable auto-derived style")
    parser.set_defaults(auto_style=None)
    # Output
    parser.add_argument("--output-dir")
    parser.add_argument("--no-save-panels", action="store_true")
    # Scenario
    parser.add_argument("--panels", type=int)
    parser.add_argument("--genre")
    parser.add_argument("--setting")
    parser.add_argument("--episode-title")
    parser.add_argument("--tone")
    parser.add_argument("--protagonist")
    parser.add_argument("--antagonist")
    parser.add_argument("--randomize", action="store_true", help="Randomize scenario elements in fallback mode")
    parser.add_argument("--no-randomize", dest="randomize", action="store_false")
    parser.set_defaults(randomize=None)
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility (fallback mode)")
    # Validation
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--val-threshold", type=float)
    parser.add_argument("--blip-model")
    parser.add_argument("--clip-model")
    # Export
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--export-width", type=int)
    parser.add_argument("--export-max-slice-height", type=int)
    parser.add_argument("--export-overlap", type=int)
    parser.add_argument("--export-format", choices=["png","jpg"])
    parser.add_argument("--export-quality", type=int)
    parser.add_argument("--export-subdir")
    return parser.parse_args()


def update_config_from_args(config: Config, args: argparse.Namespace) -> Config:
    a = {k: v for k, v in vars(args).items() if v is not None}
    # Model
    if "sd_model" in a: config.model.stable_diffusion_model = a["sd_model"]
    if "llm_model" in a: config.model.llm_model = a["llm_model"]
    if "device" in a: config.model.device = a["device"]
    if "hf_token" in a: config.model.REMOVED_TOKENtoken = a["hf_token"]
    # Generation
    if "guidance_scale" in a: config.generation.guidance_scale = a["guidance_scale"]
    if "num_steps" in a: config.generation.num_inference_steps = a["num_steps"]
    if "temperature" in a: config.generation.temperature = a["temperature"]
    # Style
    if "panel_width" in a: config.style.panel_width = a["panel_width"]
    if "panel_gap" in a: config.style.panel_gap = a["panel_gap"]
    if "panel_width_jitter" in a: config.style.panel_width_jitter = a["panel_width_jitter"]
    if "canvas_bg" in a: config.style.canvas_bg = a["canvas_bg"]
    if "panel_border_width" in a: config.style.panel_border_width = a["panel_border_width"]
    if "panel_border_color" in a: config.style.panel_border_color = a["panel_border_color"]
    if "square_panels" in a and a["square_panels"] is not None:
        config.style.square_panels = a["square_panels"]
    if "square_size" in a:
        config.style.square_size = a["square_size"]
    if "square_fill_color" in a:
        config.style.square_fill_color = a["square_fill_color"]
    if "same_panel_size" in a and a["same_panel_size"] is not None:
        config.style.same_panel_size = a["same_panel_size"]
    if "auto_panel_width" in a and a["auto_panel_width"] is not None:
        config.style.auto_panel_width = a["auto_panel_width"]
    if "apply_dramatic_lighting" in a:
        config.style.apply_dramatic_lighting = a["apply_dramatic_lighting"]
    if "lighting_style" in a:
        config.style.lighting_style = a["lighting_style"]
    if "lighting_intensity" in a:
        config.style.lighting_intensity = a["lighting_intensity"]
    if "auto_style" in a and a["auto_style"] is not None:
        config.style.auto_style = a["auto_style"]
    # Output
    if "output_dir" in a: 
        config.output.output_dir = a["output_dir"]
    if "no_save_panels" in a:
        config.output.save_individual_panels = not a["no_save_panels"]
    # Scenario
    if "panels" in a: config.scenario.panels = a["panels"]
    if "genre" in a: config.scenario.genre = a["genre"]
    if "setting" in a: config.scenario.setting = a["setting"]
    if "episode_title" in a: config.scenario.episode_title = a["episode_title"]
    if "tone" in a: config.scenario.tone = a["tone"]
    if "protagonist" in a: config.scenario.protagonist = a["protagonist"]
    if "antagonist" in a: config.scenario.antagonist = a["antagonist"]
    if "randomize" in a and a["randomize"] is not None:
        config.scenario.randomize = a["randomize"]
    if "seed" in a:
        config.scenario.random_seed = a["seed"]
    # Validation
    if "validate" in a: config.validation.enabled = a["validate"]
    if "val_threshold" in a: config.validation.similarity_threshold = a["val_threshold"]
    if "blip_model" in a: config.validation.blip_model = a["blip_model"]
    if "clip_model" in a: config.validation.clip_model = a["clip_model"]
    # Export
    if "export" in a: config.export.enabled = a["export"]
    if "export_width" in a: config.export.width = a["export_width"]
    if "export_max_slice_height" in a: config.export.max_slice_height = a["export_max_slice_height"]
    if "export_overlap" in a: config.export.overlap = a["export_overlap"]
    if "export_format" in a: config.export.format = a["export_format"]
    if "export_quality" in a: config.export.quality = a["export_quality"]
    if "export_subdir" in a: config.export.output_subdir = a["export_subdir"]
    return config
