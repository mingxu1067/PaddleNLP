export PYTHONPATH=/paddle/sparsity/paddle_nlp_glue/PaddleNLP:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0

export TASK_NAME=cola

python3.8 -u ./run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased-prelayernorm \
    --task_name $TASK_NAME \
    --max_seq_length 128 \
    --batch_size 64   \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 20 \
    --max_steps 60 \
    --use_amp true \
    --scale_loss 32768.0 \
    --use_pure_fp16 false \
    --output_dir ./tmp/$TASK_NAME/ \
    --load_dir ./glue_dense_start/$TASK_NAME/