#!/bin/bash

# script to run benchmarked experiments
export CUDA_VISIBLE_DEVICES=0
dataset_dirs=('CHQ-Summ')
pretrained_model_path_list=('microsoft/prophetnet-large-uncased' 't5-large' 'facebook/bart-large' 'google/pegasus-large')

EPOCHS=10
for dataset_dir in "${dataset_dirs[@]}"
do
  for pretrained_model_path in "${pretrained_model_path_list[@]}"
  do
    if grep -q "prophetnet" <<< "$pretrained_model_path"; then
      model_name="prophetnet"
      learning_rate_init=3e-5
      gradient_accum_steps=1
      echo "prophetnet model"
    fi

    if grep -q "t5" <<< "$pretrained_model_path"; then
      model_name="t5"
      learning_rate_init=3e-3
      gradient_accum_steps=8
      echo "t5 model"
    fi

    if grep -q "bart" <<< "$pretrained_model_path"; then
      model_name="bart"
      learning_rate_init=3e-5
      gradient_accum_steps=1
      echo "bart model"
    fi

    if grep -q "pegasus" <<< "$pretrained_model_path"; then
      model_name="pegasus"
      learning_rate_init=3e-5
      gradient_accum_steps=1
      echo "pegasus model"
    fi
    export OUTPUT_DIR_NAME=${dataset_dir}
    export CURRENT_DIR=/path/to/the/transformers/examples/seq2seq
    export OUTPUT_DIR=${CURRENT_DIR}/models/${model_name}/${OUTPUT_DIR_NAME}
    export DATASET_DIR=${CURRENT_DIR}/dataset/${dataset_dir}
    export RESULTS_DIR=${CURRENT_DIR}/results/${dataset_dir}/${model_name}

    echo $dataset_dir
    echo $OUTPUT_DIR
    echo $DATASET_DIR


    # Make output directory if it doesn't exist
    mkdir -p $OUTPUT_DIR
    mkdir -p $RESULTS_DIR


    export PYTHONPATH=${CURRENT_DIR}:"${PYTHONPATH}"


    max_src_length=300
    max_tgt_length=50

    ### training

    export CUDA_LAUNCH_BLOCKING=1
    python -m torch.distributed.launch  --nproc_per_node 1 finetune_trainer.py \
    --data_dir=$DATASET_DIR \
    --model_name_or_path=$pretrained_model_path  \
    --learning_rate=$learning_rate_init --max_source_length=$max_src_length \
    --max_target_length=$max_tgt_length --n_val 1000 --val_max_target_length=$max_tgt_length \
    --test_max_target_length=$max_tgt_length --fp16 --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4  --output_dir $OUTPUT_DIR --do_train --num_train_epochs $EPOCHS \
    --overwrite_output_dir --load_best_model_at_end  --evaluation_strategy epoch \
    --gradient_accumulation_steps=$gradient_accum_steps


    max_val_src_length=$max_src_length
    max_val_tgt_length=$max_tgt_length


    ### validate
    export PYTHONPATH=${CURRENT_DIR}:"${PYTHONPATH}"
    python generate_and_evaluate.py \
      --MODEL_PATH $OUTPUT_DIR \
      --DATASET_PATH $DATASET_DIR \
      --MAX_LEN $max_val_src_length \
      --SUMMARY_LEN $max_val_tgt_length \
      --OUTPUT_PATH $RESULTS_DIR>${RESULTS_DIR}/results.txt \
      --MODE val \
      --MODEL_NAME $model_name

    ### generate
    export PYTHONPATH=${CURRENT_DIR}:"${PYTHONPATH}"
    python generate_and_evaluate.py \
      --MODEL_PATH $OUTPUT_DIR \
      --DATASET_PATH $DATASET_DIR \
      --MAX_LEN $max_val_src_length \
      --SUMMARY_LEN $max_val_tgt_length \
      --OUTPUT_PATH $RESULTS_DIR>>${RESULTS_DIR}/results.txt \
      --MODE test \
      --MODEL_NAME $model_name
    done
done
