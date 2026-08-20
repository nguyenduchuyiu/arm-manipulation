"""Convert a URDF file to a MuJoCo MJCF (XML) file using mujoco_warp / mjpython.

Requires MuJoCo >= 3.1.0 which ships with the URDF compiler.
Usage:
    python scripts/convert_urdf.py assets/robot/robot.urdf assets/robot/robot.xml
"""
import argparse

import mujoco


def main():
    parser = argparse.ArgumentParser(description="Convert URDF to MJCF")
    parser.add_argument("urdf_path", help="Path to input URDF file")
    parser.add_argument("mjcf_path", help="Path to output MJCF XML file")
    args = parser.parse_args()

    # MuJoCo can compile a URDF directly into an MjModel, then export it.
    model = mujoco.MjModel.from_xml_path(args.urdf_path)
    mujoco.mj_saveLastXML(args.mjcf_path, model)
    print(f"Converted {args.urdf_path} -> {args.mjcf_path}")


if __name__ == "__main__":
    main()