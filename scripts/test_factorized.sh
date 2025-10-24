export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

model_name=CATSF

python -u run.py \
  --is_training 0 \
  --root_path dataset/nift/ \
  --data_path NIFTY_COMMODITIES_30T_VALID.csv \
  --model_id nift_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 512 \
  --pred_len 96 \
  --d_layers 3 \
  --dec_in 4 \
  --des 'Exp' \
  --itr 1 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 32 \
  --QAM_end 0.2 \
  --batch_size 16 \
  --target 'close' \
  --use_gpu 1

# python -u run.py \
#   --is_training 0 \
#   --root_path dataset/nift/ \
#   --data_path NIFTY_COMMODITIES_30T_VALID.csv \
#   --model_id nift_96_192\
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 192 \
#   --d_layers 3 \
#   --dec_in 4 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model 256 \
#   --d_ff 512 \
#   --n_heads 32 \
#   --QAM_end 0.3 \
#   --target 'close' \
#   --batch_size 64

# python -u run.py \
#   --is_training 0 \
#   --root_path dataset/nift/ \
#   --data_path NIFTY_COMMODITIES_30T_VALID.csv \
#   --model_id nift_96_336 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 336 \
#   --d_layers 3 \
#   --dec_in 4 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model 256 \
#   --d_ff 512 \
#   --n_heads 32 \
#   --QAM_end 0.5 \
#   --target 'close' \
#   --batch_size 64

# python -u run.py \
#   --is_training 0 \
#   --root_path dataset/nift/ \
#   --data_path NIFTY_COMMODITIES_30T_VALID.csv \
#   --model_id nift_96_720 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 720 \
#   --d_layers 3 \
#   --dec_in 4 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model 256 \
#   --d_ff 512 \
#   --n_heads 32 \
#   --QAM_end 0.7 \
#   --target 'close' \
#   --batch_size 64

