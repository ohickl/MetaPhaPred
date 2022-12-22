import numpy as np
import keras
from Bio import SeqIO

file_path = './data/rumen_phage.fasta'

def read_fasta(fasta_path):
    fasta = SeqIO.parse(fasta_path, "fasta")

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

def get_data(seq,dict):
    wordseq=sentence2word(seq)
    index=[]
    for i in range(len(wordseq)):
        temp = np.zeros((1,len(wordseq[i])))
        for j in range(len(wordseq[i])):
            temp[0][j] = dict.item().get(wordseq[i][j])
        index.append(temp[0])
    return index

model400 = keras.models.load_model('./model/100to400')
model800 = keras.models.load_model('./model/401to800')
model1200 = keras.models.load_model('./model/801to1200')

dict=np.load('dict.npy',allow_pickle=True)
phage_test, phage_test_R = read_fasta(file_path)
X_test = get_data(phage_test, dict)
X_test_R = get_data(phage_test_R, dict)

y_pred = list()
for i in range(len(X_test)):
    temp=X_test[i]
    temp_R=X_test_R[i]
    result_pre = list()
    a_all = 0
    while temp!=[]:
        if len(temp)>1198:
            model=model1200
            preds = model.predict([np.reshape(temp[0:1198],(1,1198)), np.reshape(temp_R[0:1198],(1,1198))], batch_size=1)
            a=1200
            a_all = a_all + 1200
            temp=temp[1198:]
            temp_R = temp_R[1198:]
        elif len(temp)>798 and len(temp)<=1198:
            model=model1200
            temp = np.pad(temp, (0, (1198 - len(temp))), 'constant')
            temp_R = np.pad(temp_R, (0, (1198 - len(temp_R))), 'constant')
            preds = model.predict([np.reshape(temp,(1,1198)), np.reshape(temp_R,(1,1198))], batch_size=1)
            a=1200
            a_all = a_all + 1200
            temp = []
        elif len(temp)>398 and len(temp)<=798:
            model=model800
            temp = np.pad(temp, (0, (798 - len(temp))), 'constant')
            temp_R = np.pad(temp_R, (0, (798 - len(temp_R))), 'constant')
            preds = model.predict([np.reshape(temp,(1,798)), np.reshape(temp_R,(1,798))], batch_size=1)
            a=800
            a_all = a_all + 800
            temp = []
        else:
            model=model400
            temp = np.pad(temp, (0, (398 - len(temp))), 'constant')
            temp_R = np.pad(temp_R, (0, (398 - len(temp_R))), 'constant')
            preds = model.predict([np.reshape(temp,(1,398)), np.reshape(temp_R,(1,398))], batch_size=1)
            a=400
            a_all = a_all + 400
            temp = []
        result_pre.append(a * preds)
    result = np.sum(np.array(result_pre) / a_all)
    y_pred.append(result)

fasta = SeqIO.parse(file_path, "fasta")
seqfile = open('output.txt', "w")
seqfile.write('name' + '\t' + 'length' + '\t' + 'score' + '\n')
for idx, entry in enumerate(fasta, start=0):
    seqfile.write(str(entry.id) + '\t' + str(len(entry)) + '\t' + str(y_pred[idx]) + '\n')
seqfile.close()
