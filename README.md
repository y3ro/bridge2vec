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

## Funding acknowledgments

Yerai Doval has been supported by the Spanish Ministry of Economy, Industry and Competitiveness (MINECO) through the ANSWER-ASAP project (TIN2017-85160-C2-2-R); by the Spanish State Secretariat for Research, Development and Innovation (which belongs to MINECO) and the European Social Fund (ESF) through a FPI fellowship (BES-2015-073768) associated with TELEPARES project (FFI2014-51978-C2-1-R); and by the Xunta de Galicia through TELGALICIA research network (ED431D 2017/12). The work of Jesús Vilares and Carlos Gómez-Rodríguez has also been funded by MINECO through the ANSWER-ASAP project (TIN2017-85160-C2-1-R in this case); and by Xunta de Galicia through a Group with Potential for Growth grant (ED431B 2017/01), a Competitive Reference Group grant (ED431C 2020/11), and a Remarkable Research Centre grant for the CITIC research centre (ED431G/01), the latter co-funded by EU with ERDF funding. Finally, Carlos Gómez-Rodríguez has also received funding from the European Research Council (ERC), under the European Union’s Horizon 2020 research and innovation programme (FASTPARSE, Grant No. 714150).
