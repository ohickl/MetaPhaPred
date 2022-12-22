import numpy as np
from keras.layers import *
from keras.models import *
from keras.optimizers import adam_v2
from sklearn.metrics import *
from keras import backend as K
from keras import initializers
from Bio import SeqIO

dic=np.load('dict.npy',allow_pickle=True)

class AttLayer(Layer):
    def __init__(self, attention_dim):
        # self.init = initializers.get('normal')
        self.init = initializers.RandomNormal(seed=10)
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self._trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def read_fasta(fasta_path):
    # Read in file
    fasta = SeqIO.parse(fasta_path, "fasta")

    # Iterate fasta and convert to matrices
    temp = []
    temp_R = []
    for idx, entry in enumerate(fasta, start=1):
        entry.seq = entry.seq.upper()

        seq = entry.seq
        seq_R = entry.seq.reverse_complement()

        temp.append(str(seq))
        temp_R.append(str(seq_R))

    matrix = np.array(temp)
    matrix_R = np.array(temp_R)
    return matrix, matrix_R

def sentence2word(str_set):
    word_seq=[]
    for sr in str_set:
        tmp=[]
        for i in range(len(sr)-2):
            if ('^[ACGT]+$' in sr[i:i + 3]):
                tmp.append('null')
            else:
                tmp.append(sr[i:i+3])
        word_seq.append(tmp)
    return word_seq

def get_data(seq,dic):
    wordseq=sentence2word(seq)
    index=np.zeros((len(wordseq),1198))
    for i in range(len(wordseq)):
        for j in range(len(wordseq[i])):
            index[i][j] = dic.item().get(wordseq[i][j])
    return index

def get_output(input_layer, hidden_layers):
    output = input_layer
    for hidden_layer in hidden_layers:
        output = hidden_layer(output)
    return output

PHAGE, PHAGE_R = read_fasta(r'./data/phage_train_801to1200.fasta')
BAC, BAC_R = read_fasta(r'./data/bac_train_801to1200.fasta')
X_phage = get_data(PHAGE, dic)
X_phage_R = get_data(PHAGE_R, dic)
X_bac = get_data(BAC, dic)
X_bac_R = get_data(BAC_R, dic)
X = np.concatenate((X_phage, X_bac))
y = np.concatenate((np.ones(len(X_phage)), np.zeros(len(X_bac))))
del X_phage, X_bac
X_R = np.concatenate((X_phage_R, X_bac_R))
del X_phage_R, X_bac_R

NB_WORDS = 65
EMBEDDING_DIM = 100
embedding_matrix = np.load('embedding_matrix.npy')

forward_input = Input(shape=(1198,))
reverse_input = Input(shape=(1198,))
hidden_layers = [
    Embedding(NB_WORDS, EMBEDDING_DIM, weights=[embedding_matrix], trainable=True),
    Conv1D(filters = 512,kernel_size = 40,activation='relu'),
    MaxPooling1D(20,20),
    Dropout(0.5),
    Bidirectional(LSTM(32,return_sequences=True)),
    AttLayer(32),
    Dense(1, activation='sigmoid')
]

forward_output = get_output(forward_input, hidden_layers)
reverse_output = get_output(reverse_input, hidden_layers)
output = Average()([forward_output, reverse_output])
model = Model(inputs=[forward_input, reverse_input], outputs=output)

model.summary()
model.compile(adam_v2.Adam(learning_rate=0.001), 'binary_crossentropy', metrics=['accuracy'])

history = model.fit(x=[X,X_R], y=y,batch_size=256, epochs=15,shuffle=True)
model.save('model/801to1200/')

PHAGE_test, PHAGE_test_R = read_fasta(r'./data/phage_test_801to1200.fasta')
BAC_test, BAC_test_R = read_fasta(r'./data/bac_test_801to1200.fasta')
X_phage_test = get_data(PHAGE_test, dic)
X_phage_test_R = get_data(PHAGE_test_R, dic)
X_bac_test = get_data(BAC_test, dic)
X_bac_test_R = get_data(BAC_test_R, dic)
y_test = np.concatenate((np.ones(len(X_phage_test)), np.zeros(len(X_bac_test))))
X_test = np.concatenate((X_phage_test, X_bac_test))
del X_phage_test, X_bac_test
X_test_reverse = np.concatenate((X_phage_test_R, X_bac_test_R))
del X_phage_test_R, X_bac_test_R

y_pred = model.predict([X_test, X_test_reverse])

mcc=matthews_corrcoef(y_test, y_pred.round())
print('mcc=' + str(mcc))
acc = accuracy_score(y_test, y_pred.round())
print('acc=' + str(acc))
recall = recall_score(y_test, y_pred.round())
print('recall=' + str(recall))
precision = precision_score(y_test, y_pred.round())
print('precision=' + str(precision))
f1 = f1_score(y_test, y_pred.round())
print('f1_score=' + str(f1))
fpr, tpr, thresholds = roc_curve(y_test, y_pred.round(), pos_label=1)
auroc = auc(fpr, tpr)
print('auroc=' + str(auroc))