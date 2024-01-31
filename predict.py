import logging
import numpy as np
import keras
from Bio import SeqIO
import os
import time
from multiprocessing import Pool
import tensorflow as tf

# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

file_path = snakemake.input['contigs']
prediction_path = snakemake.output['prediction']
LOG_PATH = snakemake.log[0]
cores = min(snakemake.threads, 32)

def log_or_print(message, log_path=LOG_PATH):
    if log_path:
        logging.info(message)
    else:
        print(message)


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



def read_fasta_in_batches(fasta_path, dict, batch_size=10000):
    fasta = SeqIO.parse(fasta_path, "fasta")

    temp = []
    temp_R = []
    for entry in fasta:
        entry.seq = entry.seq.upper()

        seq = str(entry.seq)
        seq_R = str(entry.seq.reverse_complement())

        temp.append(seq)
        temp_R.append(seq_R)

        # Check if the batch is full
        if len(temp) == batch_size:
            X_test = get_data(temp, dict)
            X_test_R = get_data(temp_R, dict)
            yield X_test, X_test_R
            temp = []
            temp_R = []

    # Yield any remaining sequences if they don't fill a complete batch
    if temp:
        X_test = get_data(temp, dict)
        X_test_R = get_data(temp_R, dict)
        yield X_test, X_test_R


def process_seq_data_batch(X_test, X_test_R, m400_path, m800_path, m1200_path):
    y_pred_batch = []

    model400 = keras.models.load_model(m400_path)
    model800 = keras.models.load_model(m800_path)
    model1200 = keras.models.load_model(m1200_path)

    for i in range(len(X_test)):
        temp=X_test[i]
        temp_R=X_test_R[i]
        result_pre = list()
        a_all = 0
        while len(temp) > 0:
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
        y_pred_batch.append(result)

    return y_pred_batch


def process_chunk(data):
    X_chunk, X_R_chunk, model400, model800, model1200 = data
    return process_seq_data_batch(X_chunk, X_R_chunk, model400, model800, model1200)

def process_batch_in_parallel(X_batch, X_batch_R, model400, model800, model1200, num_chunks=4):
    # Divide the batch into chunks
    chunk_size = len(X_batch) // num_chunks
    chunks = [(X_batch[i:i + chunk_size], X_batch_R[i:i + chunk_size], model400, model800, model1200) for i in range(0, len(X_batch), chunk_size)]

    # Process chunks in parallel
    with Pool(num_chunks) as pool:
        results = pool.map(process_chunk, chunks)

    # Combine results from all chunks
    y_pred_batch = [item for sublist in results for item in sublist]

    return y_pred_batch

###
# Setting up logging
if LOG_PATH:
    logging.basicConfig(filename=LOG_PATH, level=logging.INFO)

log_or_print("Starting MetaPhaPred.")


script_dir = os.path.dirname(os.path.abspath(__file__))

num_sequences = sum(1 for _ in SeqIO.parse(file_path, "fasta"))
log_or_print(f"Number of sequences: {num_sequences}")

log_or_print("Loading models ...")

# Model paths
m400_path = os.path.join(script_dir, 'model/100to400')
m800_path = os.path.join(script_dir, 'model/401to800')
m1200_path = os.path.join(script_dir, 'model/801to1200')

log_or_print("Loading data ...")
dict = np.load(os.path.join(script_dir, 'dict.npy'), allow_pickle=True)

log_or_print("Predicting ...")
y_pred = []

# Start timer for progress logging
timer = time.time()
total_time_min = 0

# X_test and X_test_R are now processed in batches
for X_batch, X_batch_R in read_fasta_in_batches(file_path, dict, 25000):
    y_pred.extend(process_batch_in_parallel(X_batch, X_batch_R, m400_path, m800_path, m1200_path, num_chunks=cores))


    # Calculate time passed since last log
    time_passed = round(time.time() - timer)
    total_time_min += round(time_passed / 60, 1)
    log_or_print(f"Predicted {len(y_pred)} of {num_sequences} sequences in {time_passed}s (total time: {total_time_min}min).")
    timer = time.time()

log_or_print("Writing prediction to file ...")
fasta = SeqIO.parse(file_path, "fasta")
seqfile = open(prediction_path, "w")
seqfile.write('name' + '\t' + 'length' + '\t' + 'score' + '\n')
for idx, entry in enumerate(fasta, start=0):
    seqfile.write(str(entry.id) + '\t' + str(len(entry)) + '\t' + str(y_pred[idx]) + '\n')
seqfile.close()

total_time_min = round(total_time_min / 60, 1)
log_or_print(f"MetaPhaPred finished successfully in {total_time_min}min.")
