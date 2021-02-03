# Recurrent Image Annotation With Explicit Inter-Label Dependencies

### In ECCV 2020 [[pdf]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740188.pdf)

## Overview

![Multi-Order-RNN Preview](https://github.com/ayushidutta/multi-order-rnn/blob/master/assets/images/multi-order-rnn-preview.png)

## Requirements

* Tensorflow 1.3
* Python 2.7

## Data and Pretrained models



## Training and Test

To train or test the model, run,

```
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
```
where, _run_opt_='train' or 'test'; _dataset_dir_ is the directory which contains the dataset _.tfrecord_ files. 

The same is provided in the bash script _run.sh_.

## Citation

If you find our work useful in your research, please cite:

## Contact

Email(First Author): ayushi.dutta@research.iiit.ac.in


