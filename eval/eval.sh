# evaliate RxnScribe_test
PORT=$((29700 + RANDOM % 1000))
echo "使用端口: $PORT 评估"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=$PORT \
    eval/eval_multigpu.py \
    --task_type "rp" \
    --model_path "eval/SFT_Model/dpollm_lr3e-7_bs64_cosine_beta01_ftx05_checkpoint-221" \
    --eval_file_path "./processed_val_data/RxnScribe_test/dev_downsampled_rxn.json" \
    --max_pixels 1000000 \
    --max_new_tokens 1024 \
    --infer_batch_size 1 \
    --pred_output_path "eval/dpo_results/RxnScribe_test/predictions.json" \
    --score_output_path "eval/dpo_results/RxnScribe_test/scores.json" \
    --per_sample_output_path "eval/dpo_results/RxnScribe_test/all_acores.json"


echo "RxnScribe_test finished. Waiting for GPU cleanup..."
sleep 60


# evaliate RobustRDP_test
PORT=$((29700 + RANDOM % 1000))
echo "使用端口: $PORT 评估"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=$PORT \
    eval/eval_multigpu.py \
    --task_type "rp" \
    --model_path "eval/SFT_Model/dpollm_lr3e-7_bs64_cosine_beta01_ftx05_checkpoint-221" \
    --eval_file_path "./processed_val_data/RobustRDP_test/dev_downsampled_rxn.json" \
    --max_pixels 1000000 \
    --max_new_tokens 1024 \
    --infer_batch_size 1 \
    --pred_output_path "eval/dpo_results/RobustRDP_test/predictions.json" \
    --score_output_path "eval/dpo_results/RobustRDP_test/scores.json" \
    --per_sample_output_path "eval/dpo_results/RobustRDP_test/all_acores.json"

