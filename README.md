# Lexical Knowledge Internalization for Neural Dialog Generation

<!-- []() -->

This repo contains code needed to replicate our findings in the [ACL’2022 paper](https://arxiv.org/abs/2205.01941) as titled. Our implementation is based on [FairSeq](https://github.com/pytorch/fairseq.git).


## Setup conda environment (recommanded)
- conda create --name ki python=3.7
- conda activate ki
- conda install pytorch -c pytorch
- cd KI/
- pip install --editable ./  

## Resources 
| File Name              | Description                                                                           | Download                                                         |
| ---------------------- | ------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `knowledge_embedding.hdf5` | Pre-extracted knowledge features. *Please put this file under "data/"* | [Link](https://1drv.ms/u/s!Ai46lXT5Q9SkhFkSMgs8SxrQ0PsH?e=KcVKc3) |
| `transformer.pt`        | checkpoint to replicate results of a transformer baseline on WoW dataset        | [Link](https://1drv.ms/u/s!Ai46lXT5Q9SkhFrsxH0Z7oPr3H_Z?e=fshY17) |
| `transformer_ki.pt`          | checkpoint to replicate results of a transformer+ki model on WoW dataset    | [Link](https://1drv.ms/u/s!Ai46lXT5Q9SkhFsuCeiieROqqt_N?e=IPrqbL) |

## Example Usage 

Here we demonstrate how to run the code on Wizard of WikiPedia (WoW) dataset.

### Data format
```bash
# We have included pre-processed (BPE, knowledge retrieval) raw data from the  dataset in the repo, 
# with the following format (take train set as an example):

train.src # source utterance
train.tgt # target response
train.voken.src # knowledge associated with each token in the source utterance, each knowledge is represented using an ID, which can be used to obtain its representation. You need a retriever to generate this file (see below).
```

#### 1. Preprocess and training
```bash
# Please download the knowledge_embedding.h5py file above before training. 
bash run_wow.sh
```

#### 2. Evaluate
```bash
bash generate.sh -b 5 -d data-bin/wow/ -c checkpoint_last10_avg.pt -s test -p checkpoints/wow_transformer_ki/  # inference 
bash evaluate.sh -p checkpoints/wow_transformer_ki/ -s test  # evaluate generated responses

evaluation script parameters:

-b beam size
-g gpu id to be used
-d data sir
-c checkpoint name 
-s test split {valid/test/test1} 
-p checkpoint dir
```

Run the evaluation commanda above, you are supposed to see:

|  Method | PPL  | wikiF1  | BLEU4  | ROUGE-l  | Distinc-1  | Distinc-2  | %safe  |
|---|---|---|---|---|---|---|---|
| Transformer+KI  | 51.03  | 14.78  | 2.74  | 12.95  | 5.94  | 21.18  | 35.42  |
| Transformer  | 49.92  | 13.56  | 2.33  | 12.88  | 4.13  |  12.71 | 59.19  |
 
> | Run bash run_baseline.sh to get results for the transformer baseline.

**Notes:** These numbers are slightly different from those reported in the paper, since the experiments are replicated on different machines and python environments. To replicate results in the paper, you can download the trained checkpoints from the links above. 



## Retriever

[The code for training and inference of retriever will be released in another repo.](https://github.com/LividWo/)

>I cannot spare hands to clean these codes recently, but if you need them in your work, please do not hesitate to email me to get an uncleaned version (with a basic doc on how to run the exp). 

