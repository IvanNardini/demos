#!/usr/bin/env bash

echo "Building Data Collection image..."
./components/data_collection/build.sh

echo "Building Data Preparation image..."
./components/data_preparation/build.sh

echo "Building Data Validation image..."
./components/data_validation/build.sh

echo "Building Feature Engineering image..."
./components/feature_engineering/build.sh
