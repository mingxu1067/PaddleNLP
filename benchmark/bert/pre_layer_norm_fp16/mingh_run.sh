export PYTHONPATH=/paddle/sparsity/PaddleNLP:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0
python3.8 -u ./run_glue.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased-prelayernorm \
        --task_name cola \
        --max_seq_length 128 \
        --select_device gpu:0 \
        --batch_size 64   \
        --learning_rate 2e-5 \
        --num_train_epochs 15 \
        --use_amp true \
        --use_pure_fp16 true \
        --load_dir ./glue_dense_start/cola/ \
        --output_dir ./glue_need_to_remove/$name/
