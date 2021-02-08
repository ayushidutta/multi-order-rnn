# Recurrent Image Annotation With Explicit Inter-Label Dependencies

### In ECCV 2020 [[pdf]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740188.pdf)

## Overview

![Multi-Order-RNN Preview](https://github.com/ayushidutta/multi-order-rnn/blob/master/assets/images/multi-order-rnn-preview.png)

## Requirements

* Tensorflow 1.3
* Python 2.7

## Data

For training and testing the model using this code, the dataset needs to be in Tensorflow's _tfrecord_ format. The _tfrecord_ files contain the: 
- images
- annotations i.e. labels
- CNN prediction probabilities (base CNN trained with sigmoid cross entropy on the same dataset). 

To convert to _tfrecord_ format, use the script:
```
python data_utils/download_and_convert_data.py --dataset_name=coco --dataset_dir=../data/coco 
```
Please refer to the scripts in _data_utils_ for details. 

The dataset split used by us is given in the _data_ folder. The images can be obtained from the respective dataset pages. The resnet features and prediction probabilities that are used to create the _tfrecord_ files, can be downloaded from the following:

[data](https://drive.google.com/drive/folders/1kPKXx7DtnVZ4ctoKx17Gl3qiKKtfENfF?usp=sharing)

## Training and Test

To train or test the model, 

```
python runner.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=coco \
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


