# Transformer-without-Explicit-Segmentation
"Transformer without Explicit Segmentation" running on Tensor2Tensor Library.

---

## Introduction

It implements the embedding algorithm suggested in the paper "Fully Character-Level Neural Machine Translation without Explicit Segmentation".

The embedding is working with the Transformer architecture, which was suggested in the paper "Attention Is All You Need".

I'm researching which embedding variation or input-output type gets better performance currently.

## Technical Report

### 1. Comparison [output type: (sub) word-level]
The result is not quite good currently.
Solving the discrepancy betweent training loss and evaluation loss would make it get better.

![word_training_loss](pictures/comp_word_training_loss.png "word_training_loss")
![word_eval_loss](pictures/comp_word_eval_loss.png "word_eval_loss")
![word_bleu](pictures/comp_word_bleu.png "word_bleu")
![word_eval_metrics](pictures/comp_word_eval_metrics.png "word_eval_metrics")


### 2. Comparison [output type: character-level]


## Reference
* [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
* [Fully Character-Level Neural Machine Translation without Explicit Segmentation](https://arxiv.org/pdf/1610.03017.pdf)
* [Character-Aware Neural Language Models](https://arxiv.org/pdf/1508.06615.pdf)
