DATASET_DIR=../data/nuswide/
TRAIN_DIR=../data/nuswide/net-res-101/lstm_sem_multi_order
python runner.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=nuswide \
    --model=lstm_sem_multi_order \
    --dim_hidden=512 \
    --dim_embed=256 \
    --prev2out=False \
    --ctx2out=False \
    --run_opt=test \
    --batch_size=32 \
    --eval_batch_size=100 \
    --loss=sigmoid \