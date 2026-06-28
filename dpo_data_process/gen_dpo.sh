# Generate DPO data
PORT=$((29700 + RANDOM % 1000))
echo "Using port: $PORT for evaluation"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=$PORT \
    dpo_data_process/gen_dpo.py \
    --model_path "./eval/SFT_Model/pretrainllm_sftall_lr1e-5_bs4_cosine_decouple_disturb_15d_checkpoint-47700" \
    --eval_file_path "./dpo_data_process/dpo_data/train_downsampled_rxn_datasource_rel_aug.json" \
    --choose_file_path "./dpo_data_process/dpo_data/train_downsampled_llm_datasource_rel_aug.json" \
    --dpo_output_path "./dpo_data_process/dpo_data/train_downsampled_llm_dpo.json" \
    --max_pixels 1000000 \
    --max_new_tokens 4500 \
    --infer_batch_size 1