from __future__ import annotations

import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from manga_ai.config import Config
from manga_ai.demo import generate_demo


DEFAULT_SCENARIO = (
    "Kai meets Maria in an empty classroom after sunset. They speak about a promise they made years ago, "
    "and both understand that keeping it will change what happens next."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the v0.1 three-panel manga demo.")
    parser.add_argument("scenario_file", nargs="?", help="Plain text file containing the short scenario.")
    parser.add_argument("--output-dir", default="output", help="Directory for demo images and metadata.")
    parser.add_argument(
        "--image-model",
        default="@cf/runwayml/stable-diffusion-v1-5-img2img",
        help="Cloudflare Workers AI image-to-image model.",
    )
    args = parser.parse_args()

    if args.scenario_file and os.path.exists(args.scenario_file):
        with open(args.scenario_file, "r", encoding="utf-8") as f:
            scenario = f.read()
    elif args.scenario_file:
        print(f"Scenario file not found: {args.scenario_file}. Using the built-in demo scenario.")
        scenario = DEFAULT_SCENARIO
    else:
        scenario = DEFAULT_SCENARIO

    config = Config.from_env()
    page_path = generate_demo(scenario, config, output_dir=args.output_dir, image_model=args.image_model)
    print(f"Demo complete: {os.path.relpath(page_path)}")


if __name__ == "__main__":
    main()
