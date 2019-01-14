# Siamese Neural Network in Tensorflow
- Tensorflow Implementation of Siamese Neural Network for Learning Sentence Similarity
- Original Pub. : http://mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf

## motivations

- Since recurrent neural network (RNN) can be used as a medium for implementing wide-range of computation problem by tuning its weight, the original authors suggest using two RNNs to learn representations of semantically coherent sentences.
- The two RNNs shares the same weight parameters to produce similar geometric output reflects their similarity relationship
- e.g. `Why do rockets look white?` =~ `Why are rockets and boosters painted white?` the weight of network is adjusted to optimal medium to minimize manhattan distance of final hidden state for each network (more details follows)
- the language representation is derived from pre-trained word vector(similar word has represented in similar numeric values) helpful for networks to adjust their weight minimally (therefore expressively) when processing similar sentences

### program design
![Program Design](docs/diagram.png)

### data preps.
- [Quora Questions Pairs Datasets](https://www.kaggle.com/c/quora-question-pairs/)
> - Download https://www.kaggle.com/c/quora-question-pairs/data
> - Unzip to obtain `all ` folder under `siamese-nn-files` folder
> - Split train data into train and test set:
> - `split -l 200000 && mv xaa train-temp.csv && mv xab test-temp.csv`

- [GloVE - Pre-trained Word-Vector Representations](https://nlp.stanford.edu/projects/glove/)
> - Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download):
> - Download http://nlp.stanford.edu/data/glove.840B.300d.zip
> - Unzip to obtain `glove.840B.300d.txt` under `siamese-nn-files` folder

### options (constants in source code)
- `train_data_max_epoch`: define training epoch
> default: 5
- `TRAIN_SINGLE_DATA_ITER`: define training batch size
> default: 1024
- `TEST_SINGLE_DATA_ITER`: define test batch size
> default: 1024
- `RNN_DIMENSION`: define rnn dimensions and number of layers (e.g. [50, 50])
> default: [50]
- `DROPOUT_PERCENTILE`: define dropout percentile (e.g. 0.5 [reference](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/DropoutWrapper))
> default: 1 (no dropout)
- `USING_PRE_TRAINED_VECTOR`: use pre-trained GloVE word vectors
> default: True
- `USE_CHAR_EMBEDDING`: use char-level embedding in addition to word-level embedding
> default: False
- `USE_LSTM_CELL`: use LSTM cell
> default: True (False to use GRU cell)

## train and test results

### mean-squared error losses
![MSE Losses](docs/loss.png)


### accuracies
![Accuracies](docs/acc.png)


## w-w/o exp. lists
- [ ] pre-trained word embeddings
- [ ] dropout
- [ ] bidirectional
- [ ] cnn-basis
- [ ] data analysis
- [ ] hierarchical attention rnn

## reference impls.
- Keras
  - https://github.com/eliorc/Medium/blob/master/MaLSTM.ipynb
  > [Medium Article : How to predict Quora Question Pairs using Siamese Manhattan LSTM](https://medium.com/mlreview/implementing-malstm-on-kaggles-quora-question-pairs-competition-8b31b0b16a07)
