from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from ..core.io import load_detection_from_json, load_sensor_frame_from_files
from ..core.manager import GraspGeneratorManager


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="统一抓取位姿生成 Demo")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML 配置文件路径")
    parser.add_argument("--end-effector-id", default="", help="可选: 手工指定末端执行器 ID；留空时走自动策略")
    parser.add_argument("--run-all", action="store_true", help="对配置中的所有 end_effector_id 逐个运行")
    parser.add_argument("--dump-json", default="", help="可选: 将结果额外保存到 json 文件")
    return parser


def load_demo_inputs(manager: GraspGeneratorManager):
    demo_config = manager.config.get("demo", {})
    sensor_frame = load_sensor_frame_from_files(
        color_path=demo_config["color_path"],
        depth_path=demo_config["depth_path"],
        intrinsics_path=demo_config["intrinsics_path"],
        extrinsics_path=demo_config.get("extrinsics_path"),
        camera_name=demo_config.get("camera_name", "middle"),
        camera_frame_id=demo_config.get("camera_frame_id", "camera"),
        world_frame_id=demo_config.get("world_frame_id", "world"),
        metadata={"sample_name": demo_config.get("sample_name", "demo_sample")},
    )
    detection_result = load_detection_from_json(demo_config["detection_path"])
    return sensor_frame, detection_result


def run_demo(manager: GraspGeneratorManager, end_effector_ids: List[str]) -> Dict[str, dict]:
    sensor_frame, detection_result = load_demo_inputs(manager)
    outputs: Dict[str, dict] = {}
    for end_effector_id in end_effector_ids:
        result = manager.generate(sensor_frame, detection_result, end_effector_id or None)
        effective_id = end_effector_id or str(result.metadata.get("selected_end_effector_id", "auto"))
        outputs[effective_id] = result.to_dict()
    return outputs


def main() -> None:
    args = build_parser().parse_args()
    manager = GraspGeneratorManager.from_yaml(args.config)

    if args.run_all:
        end_effector_ids = manager.available_end_effectors()
    else:
        end_effector_ids = [args.end_effector_id]

    results = run_demo(manager, end_effector_ids)
    result_key = end_effector_ids[0] if args.run_all or args.end_effector_id else next(iter(results.keys()))
    payload = results if args.run_all else results[result_key]
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)

    if args.dump_json:
        output_path = Path(args.dump_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
