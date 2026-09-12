#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

BATCH_NUM=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --batch) BATCH_NUM="$2"; shift ;;
        -h|--help) 
            echo "Usage: ./run_pipeline.sh [--batch N]"
            echo "If --batch is not specified, batches 1 to 4 will be processed."
            exit 0 
            ;;
        *) 
            echo "Unknown parameter: $1"
            exit 1 
            ;;
    esac
    shift
done

# Determine which batches to process
if [ -z "$BATCH_NUM" ]; then
    echo "No batch specified. Starting full pipeline for batches 1, 2, 3, and 4."
    BATCHES=(1 2 3 4)
else
    echo "Starting pipeline for single batch: $BATCH_NUM"
    BATCHES=($BATCH_NUM)
fi

# Pipeline execution loop
for b in "${BATCHES[@]}"; do
    echo ""
    echo "========================================"
    echo " STARTING PROCESS - BATCH $b"
    echo "========================================"
    
    echo ">>> [1/3] Extracting MeSH concepts (Semantic Concept Linking)..."
    python src/01_mesh_grounding.py --batch $b
    
    echo ">>> [2/3] Building Prompts and LLM Reranking..."
    python src/02_llm_reranker.py --batch $b
    
    echo ">>> [3/3] Evaluating metrics (MAP@3, MAP@5, MAP@10)..."
    python src/03_evaluate.py --batch $b
    
    echo "========================================"
    echo " BATCH $b COMPLETED SUCCESSFULLY"
    echo "========================================"
done

echo "All processes finished."