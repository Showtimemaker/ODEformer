#export CUDA_VISIBLE_DEVICES=2

model_name1=ODEformer
model_name2=Autoformer
model_name3=Reformer
model_name4=Informer
model_name5=Pyraformer

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --data_path qld2020.csv\
  --model $model_name1 \
  --data ETTm1 \
  --features S \
  --seq_len 6  \
  --label_len 6 \
  --pred_len 1 \
  --e_layers 2 \
  --d_layers 3 \

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --data_path qld2020.csv\
  --model $model_name1 \
  --data ETTm1 \
  --features S \
  --seq_len 6  \
  --label_len 6 \
  --pred_len 2 \
  --e_layers 2 \
  --d_layers 3 \
  
  python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --data_path qld2020.csv\
  --model $model_name1 \
  --data ETTm1 \
  --features S \
  --seq_len 6  \
  --label_len 6 \
  --pred_len 4 \
  --e_layers 2 \
  --d_layers 3 \
  
  python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --data_path qld2020.csv\
  --model $model_name1 \
  --data ETTm1 \
  --features S \
  --seq_len 6  \
  --label_len 6 \
  --pred_len 6 \
  --e_layers 2 \
  --d_layers 3 \
  
  python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --data_path qld2020.csv\
  --model $model_name1 \
  --data ETTm1 \
  --features S \
  --seq_len 6  \
  --label_len 6 \
  --pred_len 8 \
  --e_layers 2 \
  --d_layers 3 \
  
    python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --data_path qld2020.csv\
  --model $model_name1 \
  --data ETTm1 \
  --features S \
  --seq_len 6  \
  --label_len 6 \
  --pred_len 12 \
  --e_layers 2 \
  --d_layers 3 \
