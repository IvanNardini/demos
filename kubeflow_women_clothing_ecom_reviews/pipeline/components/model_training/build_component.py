#!/usr/bin/env python3

from pathlib import Path
import os
import kfp.components as cpt
from component.run_train import run_train
from functools import partial, update_wrapper
import argparse

REGISTRY = "docker.io/in92"
IMAGE_NAME = "model_trainer:latest"
COMP_NAME = "component.yaml"
MODELS = ['logit', 'dtree', 'rf', 'gb', 'xgb', 'lightgb']

def run_build_component(args):
    out_components_dir = args.output_component_dir

    for model in MODELS:
        model_component_dir = os.path.join(out_components_dir, model)
        p = Path(model_component_dir)
        if not p.exists():
            os.makedirs(model_component_dir, exist_ok=True)

        component = cpt.func_to_container_op(run_train,
                                             base_image=f'{REGISTRY}/{IMAGE_NAME}',
                                             output_component_file=f'{model_component_dir}/{COMP_NAME}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run component builder")
    parser.add_argument('--output-component-dir',
                        default='../../../deliverables/components/model_training')
    args = parser.parse_args()
    run_build_component(args=args)
