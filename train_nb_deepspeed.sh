# Modern version, based on Torchrun
# max_steps="800"
# warmup_steps="200"
# added adam 8 bit 
# see: https://github.com/huggingface/community-events/blob/main/whisper-fine-tuning- event/README.md#tips-and-tricks

LOCAL_RANK=0,1 CUDA_VISIBLE_DEVICES=0,1 \
deepspeed \
    run_speech_recognition_seq2seq.py \
    --deepspeed="ds_config.json" \
	--model_name_or_path="openai/whisper-large-v2" \
	--dataset_name="luigisaetta/atco2_atcosim" \
    --language="en" \
	--train_split_name="train" \
	--eval_split_name="test" \
	--max_steps="1000" \
	--output_dir="/mnt/output" \
	--per_device_train_batch_size="4" \
	--per_device_eval_batch_size="4" \
	--logging_steps="25" \
	--learning_rate="1e-5" \
	--warmup_steps="200" \
	--evaluation_strategy="steps" \
	--eval_steps="50" \
	--save_strategy="steps" \
	--save_steps="50" \
	--generation_max_length="225" \
	--length_column_name="input_length" \
	--max_duration_in_seconds="30" \
	--text_column_name="sentence" \
	--freeze_feature_encoder="False" \
	--gradient_checkpointing \
    --gradient_accumulation_steps=4 \
	--fp16 \
	--overwrite_output_dir \
	--do_train \
	--do_eval \
	--predict_with_generate \
	--use_auth_token\