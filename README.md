**This code is for paper `Hierarchical Transformers for Multi-Document Summarization` in ACL2019**

**Python version**: This code is in Python3.6

**Package Requirements**: pytorch tensorboardX pyrouge

Some codes are borrowed from ONMT(https://github.com/OpenNMT/OpenNMT-py)

**Trained Model**: Please use this url for a trained model https://drive.google.com/open?id=1suMaf7yBZPSBtaJKLnPyQ4uZNcm0VgDz



## Data Preparation:

The raw version of the WikiSum dataset can be built following steps in https://github.com/tensorflow/tensor2tensor/tree/5acf4a44cc2cbe91cd788734075376af0f8dd3f4/tensor2tensor/data_generators/wikisum

Since the raw version of the WikiSum dataset is huge (~300GB), we only provide the **ranked version** of the dataset:
https://drive.google.com/open?id=1iiPkkZHE9mLhbF5b39y3ua5MJhgGM-5u

* The dataset is ranked by the ranker as described in the paper.
* The top-40 paragraphs for each instance is extracted.
* Extract the `.pt` files into one directory.

The **sentencepiece vocab file** is at:
https://drive.google.com/open?id=1zbUILFWrgIeLEzSP93Mz5JkdVwVPY9Bf

## Model Training:

To train with the default setting as in the paper:
```
python train_abstractive.py -data_path DATA_PATH/WIKI -mode train -batch_size 10500 -seed 666 -train_steps 1000000 -save_checkpoint_steps 5000  -report_every 100 -trunc_tgt_ntoken 400 -trunc_src_nblock 24 -visible_gpus 0,1,2,3 -gpu_ranks 0,1,2,3 -world_size 4 -accum_count 4 -dec_dropout 0.1 -enc_dropout 0.1 -label_smoothing 0.1 -vocab_path VOCAB_PATH  -model_path MODEL_PATH -accum_count 4  -log_file LOG_PATH  -inter_layers 6,7 -inter_heads 8 -hier
```
* DATA_PATH is where you put the `.pt` files
* VOCAB_PATH is where you put the sentencepiece model file
* MODEL_PATH is the directory you want to store the checkpoints
* LOG_PATH is the file you want to write training logs in


## Evaluation

To evaluated the trained model:
```
python train_abstractive.py -data_path DATA_PATH/WIKI -mode validate  -batch_size 30000 -valid_batch_size 7500 -seed 666 -trunc_tgt_ntoken 400 -trunc_src_nblock 40 -visible_gpus 0 -gpu_ranks 0 -vocab_path VOCAB_PATH  -model_path MODEL_PATH -log_file LOG_PATH  -inter_layers 6,7 -inter_heads 8 -hier -report_rouge -max_wiki 100000  -dataset test -alpha 0.4 -max_length 400
```
This will load all saved checkpoints in the model_path and calculate validation losses (this could take a long time). Then it will select the top checkpoints and generate real summaries. Rouge scores will be calculated.

