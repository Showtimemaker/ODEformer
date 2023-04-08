#export CUDA_VISIBLE_DEVICES=1

model_name1=Reformer
model_name2=Informer
model_name3=Pyraformer

#model_name3=NODEtry_split_embed
#model_name4=Pyraformer
#model_name5=NODEtry_split_embed

#python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path D:/Time-Series-Library-main \
#  --data_path qld2020.csv\
#  --model $model_name1 \
#  --data ETTm1 \
#  --features S \
#  --seq_len 6 \
#  --label_len 6 \
#  --pred_len 2 \
#  --e_layers 2 \
#  --d_layers 3 \
#
#python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#   --root_path D:/Time-Series-Library-main \
#  --data_path qld2020.csv\
#  --model $model_name1 \
#  --data ETTm1 \
#  --features S \
#  --seq_len 6 \
#  --label_len 6 \
#  --pred_len 4 \
#  --e_layers 2 \
#  --d_layers 3 \
#
#python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path D:/Time-Series-Library-main \
#  --data_path qld2020.csv\
#  --model $model_name1 \
#  --data ETTm1 \
#  --features S \
#  --seq_len 6 \
#  --label_len 6 \
#  --pred_len 6 \
#  --e_layers 2 \
#  --d_layers 3 \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path D:/Time-Series-Library-main \
  --data_path qld2021.csv\
  --model $model_name1 \
  --data ETTm1 \
  --features S \
  --seq_len 6 \
  --label_len 6 \
  --pred_len 1 \
  --e_layers 2 \
  --d_layers 3 \

#
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path D:/Time-Series-Library-main \
  --data_path qld2021.csv\
  --model $model_name1 \
  --data ETTm1 \
  --features S \
  --seq_len 6 \
  --label_len 6 \
  --pred_len 2 \
  --e_layers 2 \
  --d_layers 3 \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path D:/Time-Series-Library-main \
  --data_path qld2021.csv\
  --model $model_name2 \
  --data ETTm1 \
  --features S \
  --seq_len 6 \
  --label_len 6 \
  --pred_len 1 \
  --e_layers 2 \
  --d_layers 3 \


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path D:/Time-Series-Library-main \
  --data_path qld2020.csv\
  --model $model_name2 \
  --data ETTm1 \
  --features S \
  --seq_len 6 \
  --label_len 6 \
  --pred_len 1 \
  --e_layers 2 \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path D:/Time-Series-Library-main \
  --data_path qld2020.csv\
  --model $model_name2 \
  --data ETTm1 \
  --features S \
  --seq_len 6 \
  --label_len 6 \
  --pred_len 2 \
  --e_layers 2 \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path D:/Time-Series-Library-main \
  --data_path qld2020.csv\
  --model $model_name3 \
  --data ETTm1 \
  --features S \
  --seq_len 6 \
  --label_len 6 \
  --pred_len 1 \
  --e_layers 2 \
  --d_layers 3 \

#
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path D:/Time-Series-Library-main \
  --data_path qld2020.csv\
  --model $model_name3 \
  --data ETTm1 \
  --features S \
  --seq_len 6 \
  --label_len 6 \
  --pred_len 2 \
  --e_layers 2 \
  --d_layers 3 \

#python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path D:/Time-Series-Library-main \
#  --data_path qld2021.csv\
#  --model $model_name2 \
#  --data ETTm1 \
#  --features S \
#  --seq_len 6 \
#  --label_len 6 \
#  --pred_len 8 \
#  --e_layers 2 \
#  --d_layers 3 \
#
#
#python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path D:/Time-Series-Library-main \
#  --data_path qld2021.csv\
#  --model $model_name2 \
#  --data ETTm1 \
#  --features S \
#  --seq_len 6 \
#  --label_len 6 \
#  --pred_len 12 \
#  --e_layers 2 \
#python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path D:/Time-Series-Library-main \
#  --data_path qld2020.csv\
#  --model $model_name1 \
#  --data ETTm1 \
#  --features S \
#  --seq_len 2 \
#  --label_len 2 \
#  --pred_len 1 \
#  --e_layers 2 \
#  --d_layers 3 \
#
#python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path D:/Time-Series-Library-main \
#  --data_path qld2020.csv\
#  --model $model_name1 \
#  --data ETTm1 \
#  --features S \
#  --seq_len 4 \
#  --label_len 4 \
#  --pred_len 1 \
#  --e_layers 2 \
#  --d_layers 3 \
#
#python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path D:/Time-Series-Library-main \
#  --data_path qld2020.csv\
#  --model $model_name1 \
#  --data ETTm1 \
#  --features S \
#  --seq_len 8 \
#  --label_len 8 \
#  --pred_len 1 \
#  --e_layers 2 \
#  --d_layers 3 \

#python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path D:/Time-Series-Library-main \
#  --data_path qld2020.csv\
#  --model $model_name3 \
#  --data ETTm1 \
#  --features S \
#  --seq_len 6 \
#  --label_len 6 \
#  --pred_len 4 \
#  --e_layers 2 \
#  --d_layers 3 \
#
#python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path D:/Time-Series-Library-main \
#  --data_path qld2020.csv\
#  --model $model_name3 \
#  --data ETTm1 \
#  --features S \
#  --seq_len 6 \
#  --label_len 6 \
#  --pred_len 6 \
#  --e_layers 2 \
#  --d_layers 3 \
#
#python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path D:/Time-Series-Library-main \
#  --data_path qld2020.csv\
#  --model $model_name3 \
#  --data ETTm1 \
#  --features S \
#  --seq_len 6 \
#  --label_len 6 \
#  --pred_len 8 \
#  --e_layers 2 \
#  --d_layers 3 \
##
##python -u run.py \
##  --task_name long_term_forecast \
##  --is_training 1 \
##  --root_path D:/Time-Series-Library-main \
##  --data_path qld2020.csv\
##  --model $model_name4 \
##  --data ETTm1 \
##  --features S \
##  --seq_len 6 \
##  --label_len 6 \
##  --pred_len 4 \
##  --e_layers 2 \
##  --d_layers 3 \
##
##python -u run.py \
##  --task_name long_term_forecast \
##  --is_training 1 \
##  --root_path D:/Time-Series-Library-main \
##  --data_path qld2020.csv\
##  --model $model_name4 \
##  --data ETTm1 \
##  --features S \
##  --seq_len 6 \
##  --label_len 6 \
##  --pred_len 6 \
##  --e_layers 2 \
##  --d_layers 3 \
##
##python -u run.py \
##  --task_name long_term_forecast \
##  --is_training 1 \
##  --root_path D:/Time-Series-Library-main \
##  --data_path qld2020.csv\
##  --model $model_name4 \
##  --data ETTm1 \
##  --features S \
##  --seq_len 6 \
##  --label_len 6 \
##  --pred_len 8 \
##  --e_layers 2 \
##  --d_layers 3 \
##
##python -u run.py \
##  --task_name long_term_forecast \
##  --is_training 1 \
##  --root_path D:/Time-Series-Library-main \
##  --data_path qld2020.csv\
##  --model $model_name5 \
##  --data ETTm1 \
##  --features S \
##  --seq_len 6 \
##  --label_len 6 \
##  --pred_len 4 \
##  --e_layers 2 \
##  --d_layers 3 \
##
##python -u run.py \
##  --task_name long_term_forecast \
##  --is_training 1 \
##  --root_path D:/Time-Series-Library-main \
##  --data_path qld2020.csv\
##  --model $model_name5 \
##  --data ETTm1 \
##  --features S \
##  --seq_len 6 \
##  --label_len 6 \
##  --pred_len 6 \
##  --e_layers 2 \
##  --d_layers 3 \
##
##python -u run.py \
##  --task_name long_term_forecast \
##  --is_training 1 \
##  --root_path D:/Time-Series-Library-main \
##  --data_path qld2020.csv\
##  --model $model_name5 \
##  --data ETTm1 \
##  --features S \
##  --seq_len 6 \
##  --label_len 6 \
##  --pred_len 8 \
##  --e_layers 2 \
##  --d_layers 3 \
