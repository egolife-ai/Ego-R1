#!/bin/bash

# EgoLife QA API Configuration
echo "🔧 Loading API Configuration..."

# =============================================================================
# Model API Configurations
# =============================================================================

# Gemini API Configuration
export GEMINI_API_KEY="your api key"  # wrq's key

# GPT-4o Configuration
export GPT4O_ENDPOINT_URL="your endpoint url"
export GPT4O_DEPLOYMENT_NAME="gpt-4o"
export GPT4O_AZURE_OPENAI_API_KEY="your api key"

# GPT-4.1 Configuration  
export GPT41_ENDPOINT_URL="your endpoint url"
export GPT41_DEPLOYMENT_NAME="gpt-4.1"
export GPT41_AZURE_OPENAI_API_KEY="your api key"

# =============================================================================
# Set Model-Specific Environment Variables
# =============================================================================

# Script to run data_gen/main.py 5 times
SCRIPT_PATH="./cott_gen/main.py"

# identity
identity="A1"
export IDENTITY=$identity

# model
MODEL="gpt-4.1"


echo "✅ API Configuration loaded successfully!"

# Create a datetime string
datetime=$(date +"%Y%m%d_%H%M%S")

# Create a directory for results
RESULT_DIR="cott_data/${identity}_${datetime}"
mkdir -p $RESULT_DIR

# Create a directory for cache
CACHE_DIR="./cache/${identity}_${datetime}"
mkdir -p $CACHE_DIR
export CACHE_DIR=$CACHE_DIR

echo "Starting to run main.py 5 times..."
echo "Using model: $MODEL"
echo "Saving the results in $RESULT_DIR"

# Run the script 5 times
for i in {1..5}; do
    echo "Run $i of 5 starting..."
    
    PYTHONPATH=$PYTHONPATH:./ python $SCRIPT_PATH --result_dir $RESULT_DIR --cache_dir $CACHE_DIR --observation_type all_actions --identity $identity
    
    if [ $? -eq 0 ]; then
        echo "Run $i completed successfully."
    else
        echo "Run $i failed with error code $?"
    fi

    sleep 10
done

echo "All runs completed."  