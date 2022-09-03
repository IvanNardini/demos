#!/usr/bin/env python3

# build_pipeline.py
# build_pipeline is a module to compile the pipeline

import kfp
import kfp.components as cmpt
import kfp.compiler as cmp
import kfp.dsl as dsl
from kfp.gcp import use_gcp_secret

import datetime
import os
import argparse

# Variables ------------------------------------------------------------------------------------------------------------

DATA_COLLECT = '../deliverables/components/data_collection/component.yaml'
DATA_PREPARE = '../deliverables/components/data_preparation/component.yaml'
DATA_VALIDATE = '../deliverables/components/data_validation/component.yaml'
FEATURE_ENGINE = '../deliverables/components/feature_engineering/component.yaml'
LOGIT_COMPONENT = '../deliverables/components/model_training/logit/component.yaml'
DTREE_COMPONENT = '../deliverables/components/model_training/dtree/component.yaml'
GB_COMPONENT = '../deliverables/components/model_training/gb/component.yaml'
RF_COMPONENT = '../deliverables/components/model_training/rf/component.yaml'
XGB_COMPONENT = '../deliverables/components/model_training/xgb/component.yaml'
LIGHT_BG_COMPONENT = '../deliverables/components/model_training/lightgb/component.yaml'
REGISTRY = "docker.io/in92"

# Components -----------------------------------------------------------------------------------------------------------

data_collection_component = cmpt.load_component_from_file(filename=DATA_COLLECT)
data_preparation_component = cmpt.load_component_from_file(filename=DATA_PREPARE)
data_validation_component = cmpt.load_component_from_file(filename=DATA_VALIDATE)
feature_engine_component = cmpt.load_component_from_file(filename=FEATURE_ENGINE)
logit_train_component = cmpt.load_component_from_file(filename=LOGIT_COMPONENT)
dtree_train_component = cmpt.load_component_from_file(filename=DTREE_COMPONENT)
gb_train_component = cmpt.load_component_from_file(filename=GB_COMPONENT)
rf_train_component = cmpt.load_component_from_file(filename=RF_COMPONENT)
xgb_train_component = cmpt.load_component_from_file(filename=XGB_COMPONENT)
lightgb_train_component = cmpt.load_component_from_file(filename=LIGHT_BG_COMPONENT)


# run_build_pipeline ---------------------------------------------------------------------------------------------------
def run_build_pipeline(args):
    out_pipe_dir = args.out_pipe_dir
    mode = args.mode

    pipe_name = f"women_clt_rev_clf_pipe_{mode}_{datetime.datetime.now().strftime('%y%m%d-%H%M%S')}.yaml"

    # TODO: Check for one to one portability with cloud, add on prem in case
    if mode == 'cloud':
        @dsl.pipeline(name="Women Clothing Reviews Classification ML Pipeline",
                      description="An example of Machine Learning Pipeline")
        def build_pipeline(config_file='config.yaml', mode=None, bucket=None):
            # Data Collection ------------------------------------------------------------------------------------------

            step_1 = (data_collection_component(config=config_file,
                                                mode=mode,
                                                bucket=bucket)
                      .set_display_name('Data Collection node')
                      .apply(use_gcp_secret('user-gcp-sa')))

            # Data Validation ------------------------------------------------------------------------------------------

            step_2 = (data_validation_component(config=config_file,
                                                mode=mode,
                                                bucket=bucket)
                      .set_display_name('Data Validation node')
                      .apply(use_gcp_secret('user-gcp-sa')))

            # Data Preparation -----------------------------------------------------------------------------------------

            step_3 = (data_preparation_component(config=config_file,
                                                 mode=mode,
                                                 bucket=bucket,
                                                 train_path=step_1.outputs['train'],
                                                 test_path=step_1.outputs['test'],
                                                 val_path=step_1.outputs['val'])
                      .set_display_name('Data Preparation node')
                      .apply(use_gcp_secret('user-gcp-sa')))

            step_3.after(step_1)

            # Feature Engineering --------------------------------------------------------------------------------------

            step_4 = (feature_engine_component(config=config_file,
                                               mode=mode,
                                               bucket=bucket,
                                               train_path=step_3.outputs['train'],
                                               test_path=step_3.outputs['test'],
                                               val_path=step_3.outputs['val'])
                      .set_memory_limit('3G')
                      .set_memory_request('3G')
                      .set_display_name('Feature Engineering node')
                      .apply(use_gcp_secret('user-gcp-sa')))

            step_4.after(step_3)

            # Model Training  ------------------------------------------------------------------------------------------

            step_5_logit = (logit_train_component(config=config_file,
                                                  mode=mode,
                                                  bucket=bucket,
                                                  train_path=step_4.outputs['train'],
                                                  test_path=step_4.outputs['test'])
                            .set_memory_limit('3G')
                            .set_memory_request('3G')
                            .set_display_name('Logistic Regression node')
                            .apply(use_gcp_secret('user-gcp-sa')))

            step_5_logit.after(step_4)

            step_5_dtree = (dtree_train_component(config=config_file,
                                                  mode=mode,
                                                  bucket=bucket,
                                                  train_path=step_4.outputs['train'],
                                                  test_path=step_4.outputs['test'])
                            .set_memory_limit('3G')
                            .set_memory_request('3G')
                            .set_display_name('Decision Tree node')
                            .apply(use_gcp_secret('user-gcp-sa')))

            step_5_dtree.after(step_4)

            step_5_gb = (gb_train_component(config=config_file,
                                            mode=mode,
                                            bucket=bucket,
                                            train_path=step_4.outputs['train'],
                                            test_path=step_4.outputs['test'])
                         .set_memory_limit('3G')
                         .set_memory_request('3G')
                         .set_display_name('Gradient Boosting node')
                         .apply(use_gcp_secret('user-gcp-sa')))

            step_5_gb.after(step_4)

            step_5_rf = (rf_train_component(config=config_file,
                                            mode=mode,
                                            bucket=bucket,
                                            train_path=step_4.outputs['train'],
                                            test_path=step_4.outputs['test'])
                         .set_memory_limit('3G')
                         .set_memory_request('3G')
                         .set_display_name('Random Forest node')
                         .apply(use_gcp_secret('user-gcp-sa')))

            step_5_rf.after(step_4)

            step_5_xgb = (xgb_train_component(config=config_file,
                                              mode=mode,
                                              bucket=bucket,
                                              train_path=step_4.outputs['train'],
                                              test_path=step_4.outputs['test'])
                          .set_memory_limit('3G')
                          .set_memory_request('3G')
                          .set_display_name('Xgb node')
                          .apply(use_gcp_secret('user-gcp-sa')))

            step_5_xgb.after(step_4)

            step_5_light_gb = (lightgb_train_component(config=config_file,
                                                       mode=mode,
                                                       bucket=bucket,
                                                       train_path=step_4.outputs['train'],
                                                       test_path=step_4.outputs['test'])
                               .set_memory_limit('3G')
                               .set_memory_request('3G')
                               .set_display_name('LightGb node')
                               .apply(use_gcp_secret('user-gcp-sa')))

            step_5_light_gb.after(step_4)

    pipeline_compiler = cmp.Compiler()
    pipeline_compiler.compile(pipeline_func=build_pipeline,
                              package_path=os.path.join(out_pipe_dir, pipe_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run pipeline builder")
    parser.add_argument('--out-pipe-dir',
                        default='../deliverables/pipeline')
    parser.add_argument('--mode',
                        default='cloud')
    args = parser.parse_args()
    run_build_pipeline(args=args)
    