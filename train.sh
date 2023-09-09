# Modern version, based on Torchrun
# max_steps="800"
# warmup_steps="200"
LOCAL_RANK=0,1 CUDA_VISIBLE_DEVICES=0,1 \
torchrun \
 	--nproc_per_node 2 \
    decompressed_artifact/run_speech_recognition_seq2seq.py \
	--model_name_or_path="openai/whisper-medium" \
	--dataset_name="luigisaetta/atco2" \
	--language="en" \
	--train_split_name="train" \
	--eval_split_name="test" \
	--max_steps="1000" \
	--output_dir="/mnt/output" \
	--per_device_train_batch_size="2" \
	--per_device_eval_batch_size="8" \
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
    --gradient_accumulation_steps=8 \
	--fp16 \
	--overwrite_output_dir \
	--do_train \
	--do_eval \
	--predict_with_generate \
	--use_auth_token\