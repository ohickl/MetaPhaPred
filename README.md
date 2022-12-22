# MetaPhaPred (Metagenomic Phage sequences Predictor)
MetaPhaPred is a deep learning method for identification of phage sequences from metagenomic data.

We first employ the word embedding technique to learn the distributed representation of DNA words and use the trained word vectors to encode DNA sequences. This step can effectively extracts the latent relationship between different DNA words. Then the embedded sequences are fed into CNN and Bi-LSTM for feature extraction. Finally, we use an attention layer to enhance the contribution of important features.

We use the model trained by 100-400 bp sequences for predicting sequences shorter than 400 bp, the model trained by 401-800 bp sequences for predicting sequences of the length 401-800 bp, the model trained by 801-1200 bp sequences for predicting sequences of the length 801-1200 bp. For sequences longer than 1200 bp, we split the sequence into non-overlapping subsequences shorter than 1200 bp, then use the corresponding model to predict each subsequence. The final prediction score of the sequence is the weighted average score of each subsequence.

# Dependencies
MetaPhaPred requires Python 3.8 with the packages of keras 2.8.0, numpy, Biopython and sklearn.

# Usage
The input of MetaPhaPred is the fasta file containing the sequences to predict. To make a prediction, change the 'file_path' in `predict.py` line 5 and run `predict.py`. The result is recorded in `output.txt`, containing the length and predicted score for each of the input sequences. The predicted score indicates the probability that the given metagenomic sequence is a phage.

To train new models using customized dataset, first, change the filepath of the dataset for training and test in `train.py` line 100-101 and 139-140, and run `train.py` to train the model. 