# Notebooks

This folder contains several Jupyter notebooks that demonstrate various features and functionalities of the kvpress package.
Below is a list of the notebooks along with a brief explanation of their content:

## [wikipedia_demo.ipynb](wikipedia_demo.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JNvaTKuuAHrl49dYB9-mdEH_y52Ib-NP?usp=drive_link)
This notebook introduces the kvpress package by compressing the Wikipedia article of Nvidia. 

## [expected_attention.ipynb](expected_attention.ipynb)
This notebook illustrates the usage of the `ExpectedAttentionPress` class. It explains how to compute scores based on the expected attention on future positions and demonstrates the steps involved in the process.

## [new_press.ipynb](new_press.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ld6u2OnVUpGryBGDdanjjDrf6j7TD0oA?usp=drive_link)
This notebook provides an overview on how to create a new press. It explains the underlying mechanism of key-value compression and how it can be applied to transformer models.

## [per_layer_compression_demo.ipynb](per_layer_compression_demo.ipynb)
This notebook provides a demonstration of the per-layer compression feature. It shows how to improve the overall compression ratio by applying a different compression ratio to each layer of the model.

## [speed_and_memory.ipynb](speed_and_memory.ipynb)
This notebook provides a demonstration how to measure the memory and throughput gains of the kvpress package.