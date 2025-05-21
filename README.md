# Quanv1D

**Quanv1D** is a quantum version of the classic Conv1D layer — instead of regular filters or kernels, it uses trainable quantum circuits. Our proposed layer can handle any number of input channels, has flexible kernel sizes, and lets you choose how many feature maps you want unlike older quanvolution methods.

This repo includes Quanv1D, plus our proposed model, Fully Quanvolutional Network (FQN), built entirely using Quanv1D layers.

The code is part of our paper accepted at *KDD'25*. If you use this repo or Quanv1D in your work, please consider citing it:

<pre lang="markdown"> 
@inproceedings{orka2025fqn,
  author    = {Orka, Nabil Anan and Haque, Ehtashamul and Awal, Md. Abdul and Moni, Mohammad Ali},
  title     = {Fully Quanvolutional Networks for Time Series Classification},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '25)},
  year      = {2025},
  volume    = {2},
  pages     = {12 pages}
  address   = {Toronto, ON, Canada},
  publisher = {ACM}
}
</pre>
