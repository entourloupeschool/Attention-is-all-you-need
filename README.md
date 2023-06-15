# Attention-is-all-you-need
GPT Tutorial - A Comprehensive Guide

This repository contains a Jupyter Notebook that provides a step-by-step tutorial on building a Generatively Pretrained Transformer (GPT), based on the seminal paper "Attention is All You Need" and OpenAI's GPT-2 / GPT-3 models.
About the Notebook

The Jupyter Notebook closely follows a tutorial video by the renowned AI Researcher Andrej Karpathy, wherein he discusses the key insights and mathematical tricks used in the implementation of GPT models. The tutorial also provides a comprehensive overview of the computation pipeline in these models.

You can access the tutorial video here: GPT Tutorial by Andrej Karpathy
Requirements

To run this notebook, you will need:

    Python 3.x
    Jupyter Notebook
    PyTorch
    Numpy

The self-attention mechanism, a fundamental building block of Transformer models used in Language Model Learning (LLM), such as GPT and BERT. Let's break it down:

    Attention as a Communication Mechanism: In the context of a graph, each node can be seen as aggregating information from other nodes that point to it. The amount of information each node takes from others is determined by data-dependent weights, effectively implementing the attention mechanism.

    No Notion of Space: Unlike convolutional layers that operate over spatially structured data, attention can operate over a set of vectors with no specific spatial or sequential arrangement. However, for tasks like language processing, we need to add positional information, typically using positional encodings.

    Batch Dimension Independence: Each example in the batch is processed independently of others, meaning they don't interact with or influence each other during processing.

    Encoder and Decoder Attention Blocks: In the context of Transformers, 'encoder' and 'decoder' attention blocks have slightly different behaviors. An encoder attention block allows every token to attend to every other token, while a decoder block uses masking to prevent future tokens from being attended to, which is crucial in autoregressive tasks like language modeling.

    Self-Attention vs Cross-Attention: In self-attention, the queries, keys, and values are all derived from the same input. In cross-attention, queries are derived from one source, but keys and values come from another, often used in a Transformer with both encoder and decoder.

    Scaled Attention: Scaling attention by 1/sqrt(head_size) helps maintain the variance of the weights, which can help prevent the softmax function from becoming too peaked and, as a result, gradients from becoming too small. This is essential to keep the learning process stable.