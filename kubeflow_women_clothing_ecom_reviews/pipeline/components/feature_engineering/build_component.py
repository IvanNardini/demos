#!/usr/bin/env python3

from pathlib import Path
import os
import kfp.components as cpt
from component.run_generate_features import run_generate_features
import argparse

REGISTRY = "docker.io/in92"
IMAGE_NAME = "feature_generator:latest"
COMP_NAME = "component.yaml"

def run_build_component(args):
    out_components_dir = args.output_component_dir
    p = Path(out_components_dir)
    if not p.exists():
        os.mkdir(out_components_dir)
    component = cpt.func_to_container_op(run_generate_features,
                                         base_image=f'{REGISTRY}/{IMAGE_NAME}',
                                         output_component_file=f'{out_components_dir}/{COMP_NAME}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run component builder")
    parser.add_argument('--mode',
                        default='cloud')
    parser.add_argument('--output-component-dir',
                        default='../../../deliverables/components/feature_engineering')
    args = parser.parse_args()
    run_build_component(args=args)