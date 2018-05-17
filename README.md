Transformer Variations 
===

"Variations of Transformer" running on Tensor2Tensor Library.

All experiments are run over Tensor2Tensor v1.2.9 and Tensorflow 1.4.0.
---

* [Transformer without segmentation](./Transformer_without_segmentation.md)
    * See [TransformerChrawr](./models/transformer_chrawr.py)
    * Transformer version of [Fully Character-Level Neural Machine Translation without Explicit Segmentation](https://arxiv.org/pdf/1610.03017.pdf)

* MOS
    * See [MixtureOfSoftmaxSymbolModality](./layers/modalities.py)
    * Add a config like: **hparams.target_modality="symbol:mos"**
    * T2T Implementation of [Breaking the Softmax Bottleneck](https://arxiv.org/pdf/1711.03953.pdf) 

* Fast Transformer
    * See [TransformerFast](./models/transformer_fast.py)
    * Add Encoder-Decoder attention cache, which is not implemented in T2T yet.
    * For my case, it is about 2.5 times faster than T2T base transformer model.

* Transformer with Average Attention Network
    * See [TransformerFastAan](./models/transformer_fast.py)
    * Add Encoder-Decoder attention cache
    * For my case, it is about 2.4 time faster than T2T base transformer model.
    * T2T Implementation of [Accelerating Neural Transformer via an Average Attention Network](https://arxiv.org/pdf/1805.00631.pdf)

