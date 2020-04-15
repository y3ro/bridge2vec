## Introduction

This is the code accompanying the paper [Towards robust word embeddings for noisy texts](https://arxiv.org/abs/1911.10876). It is an adaptation of the [fastText](https://github.com/facebookresearch/fastText) tool by Facebook (although an older version).

## Requirements

Compilation is carried out using a Makefile, so you will need to have a working **make** and compilers with good C++11 support, such as g++-4.7.2 or clang-3.3, or newer.
You will also need the [utf8proc](https://juliastrings.github.io/utf8proc/) library. 

## Building bridge2vec

```
$ git clone https://github.com/yeraidm/bridge2vec.git
$ cd bridge2vec
$ make
```

### Example usage

```
$ ./fasttext skipgram -input data.txt -output model
```

where `data.txt` is a training file containing `UTF-8` encoded text.
At the end of optimization the program will save two files: `model.bin` and `model.vec`.
`model.vec` is a text file containing the word vectors, one per line.
`model.bin` is a binary file containing the parameters of the model along with the dictionary and all hyper parameters.

For more information, see the original fastText's README included.

## Reference

Please cite us if you use this code in your paper:

```
@article{doval2019robust,
  title={Towards robust word embeddings for noisy texts},
  author={Yerai Doval and Jesús Vilares and Carlos Gómez-Rodríguez},
  journal={arXiv preprint arXiv:1911.10876},
  year={2019}
}
```

## License

bridge2vec is BSD-licensed.
