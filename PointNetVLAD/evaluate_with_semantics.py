import argparse
import math
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from pointnetvlad_cls import *
from loading_pointclouds_kitti import *
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from tqdm import tqdm
import time
from scipy.stats import wasserstein_distance

#params
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--positives_per_query', type=int, default=4, help='Number of potential positives in each training tuple [default: 2]')
parser.add_argument('--negatives_per_query', type=int, default=12, help='Number of definite negatives in each training tuple [default: 20]')
parser.add_argument('--batch_num_queries', type=int, default=3, help='Batch Size during training [default: 1]')
parser.add_argument('--dimension', type=int, default=256)
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
parser.add_argument('--sequence', type=str, default='08', help= 'Sequence number [default: 05]')

FLAGS = parser.parse_args()

BATCH_NUM_QUERIES = FLAGS.batch_num_queries
EVAL_BATCH_SIZE = 1
NUM_POINTS = 4096
POSITIVES_PER_QUERY= FLAGS.positives_per_query
NEGATIVES_PER_QUERY= FLAGS.negatives_per_query
GPU_INDEX = FLAGS.gpu
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

RESULTS_FOLDER="pretrained_results_D-3_T-0/with_labels/"
if not os.path.exists(RESULTS_FOLDER): os.mkdir(RESULTS_FOLDER)
output_file= RESULTS_FOLDER +'recalls_and_precisions_4096.txt'

SQ = FLAGS.sequence

KITTI_submap_dir = "Datasets/KITTI/sequences_with_labels"
model_file= "model.ckpt"

query_sets = "./KITTI_all/positive_sequence_D-3_T-0.json"
f = open(query_sets, "r")
QUERY_SETS = json.load(f)

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_NUM_QUERIES,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay     


def evaluate():
    sequences = ["00", "02", "05", "06", "07", "08"]
    avg_precisions = []  
    avg_precisions_hist = []  
    avg_recalls = []  
    avg_recalls_hist = []  
    for sq in sequences:
        LOG_DIR = 'log_fold{}/'.format(sq)
        with tf.Graph().as_default():
            with tf.device('/gpu:'+str(GPU_INDEX)):
                print("In Graph")
                query= placeholder_inputs(BATCH_NUM_QUERIES, 1, NUM_POINTS)
                positives= placeholder_inputs(BATCH_NUM_QUERIES, POSITIVES_PER_QUERY, NUM_POINTS)
                negatives= placeholder_inputs(BATCH_NUM_QUERIES, NEGATIVES_PER_QUERY, NUM_POINTS)
                eval_queries= placeholder_inputs(EVAL_BATCH_SIZE, 1, NUM_POINTS)

                is_training_pl = tf.placeholder(tf.bool, shape=())
                print(is_training_pl)

                batch = tf.Variable(0)
                bn_decay = get_bn_decay(batch)

                with tf.variable_scope("query_triplets") as scope:
                    vecs= tf.concat([query, positives, negatives],1)
                    print(vecs)                
                    out_vecs= forward(vecs, is_training_pl, bn_decay=bn_decay)
                    print(out_vecs)
                    q_vec, pos_vecs, neg_vecs= tf.split(out_vecs, [1,POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY],1)
                    print(q_vec)
                    print(pos_vecs)
                    print(neg_vecs)

                saver = tf.train.Saver()
                
            # Create a session
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
            config = tf.ConfigProto(gpu_options=gpu_options)
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.Session(config=config)


            saver.restore(sess, os.path.join(LOG_DIR, model_file))
            print("model file:", os.path.join(LOG_DIR, model_file))
            print("Model restored.")

            ops = {'query': query,
                'positives': positives,
                'negatives': negatives,
                'is_training_pl': is_training_pl,
                'eval_queries': eval_queries,
                'q_vec':q_vec,
                'pos_vecs': pos_vecs,
                'neg_vecs': neg_vecs}
            
            sq_dir = os.path.join(KITTI_submap_dir, sq)
            t1 = time.time()
            queries_labels = get_histogram(sess, ops, sq_dir) # get indices predicted by the histogram with wasserstein distance
            t2 = time.time()
            print("time: ", t2 - t1)

            feature_queries = np.load('/home/user/pointnetvlad/log_fold{}/feature_database/queries_4096_completed/{}_PV_{}.npy'.format(sq,sq,sq))
            feature_database = np.load('/home/user/pointnetvlad/log_fold{}/feature_database/database_4096_completed/{}_PV_{}.npy'.format(sq,sq,sq))
            
            avg_recall, avg_precision = get_recall(sess, ops, feature_database, feature_queries, sq)
            avg_recall_hist, avg_precision_hist = get_recall_hist(sess, ops, feature_database, feature_queries, sq, queries_labels)

            avg_precisions.append(avg_precision)
            avg_precisions_hist.append(avg_precision_hist)
            avg_recalls.append(avg_recall)
            avg_recalls_hist.append(avg_recall_hist)
            
    with open(output_file, "w") as output:
        output.write("Average Precision:\n")
        output.write(str(sequences))
        output.write("\n")
        output.write("without label information: ")
        output.write(str(avg_precisions))
        output.write("\n")
        output.write("with label information: ")
        output.write(str(avg_precisions_hist))
        # output.write("\n\n")
        # output.write("Average Recalls:\n")
        # output.write(str(sequences))
        # output.write("\n")
        # output.write("completed: ")
        # output.write(str(avg_recalls))
        # output.write("\n")
        # output.write("original: ")
        # output.write(str(avg_recalls_original))

def get_histogram(sess, ops, sq_dir):
    train_file_idxs = []
    listDir(sq_dir, train_file_idxs)
    train_file_idxs.sort()

    batch_num= BATCH_NUM_QUERIES*(1+POSITIVES_PER_QUERY+NEGATIVES_PER_QUERY)
    histo_output = []
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    
    for q_index in tqdm(range(len(train_file_idxs)//batch_num)):
        file_names=train_file_idxs[q_index*batch_num:(q_index+1)*(batch_num)]
        queries_labels=load_pc_labels(file_names)
        for query_labels in queries_labels:
            query_histogram = np.histogram(query_labels, bins)
            histo_output.append(query_histogram)
    histo_output = np.array(histo_output)
 
    # edge case 
    for q_index in tqdm(range((len(train_file_idxs)//batch_num*batch_num),len(train_file_idxs))):
        file_names=train_file_idxs[q_index*batch_num:(q_index+1)*(batch_num)]
        queries_labels=load_pc_labels(file_names)
        for query_labels in queries_labels:
            query_histogram = np.histogram(query_labels, bins)
            histo_output.append(query_histogram)
        if (histo_output.shape[0]!=0):
            histo_output=np.vstack((histo_output,query_histogram))
        else:
            histo_output=query_histogram
    print(histo_output.shape)
    print(len(train_file_idxs))
    assert histo_output.shape[0] == len(train_file_idxs)
    print("Done!")

    return histo_output

def get_recall_hist(sess, ops, database_output, queries_output, sq, queries_labels=None):
    # queries_output: latent vectors of the completed submaps from 00
    # database_output: latent vectors of the original submaps from 00

    print(len(queries_output))
    database_nbrs = KDTree(database_output)

    recalls = []
    precisions = []

    num_evaluated=0
    for i in range(len(queries_output)): # the range is 4651
        true_neighbors=QUERY_SETS[sq][str(i)] # the json file, which defines the true neighbors of all submaps in sequence 00
        anchor_labels = queries_labels[i]
        num_true_neighbors = len(true_neighbors)
        num_pred_neighbors = num_true_neighbors + 5
        ws_dists = []
        if(len(true_neighbors)==0):
            continue
        num_evaluated+=1
        
        _, indices = database_nbrs.query(np.array([queries_output[i]]),k=num_pred_neighbors)
        preds_labels = queries_labels[indices]
        preds_labels = preds_labels[0]
        indices_list = np.ndarray.tolist(indices[0])

        for pred_labels in preds_labels:
            # calculate the wasser stein distance between the query and the prediction
            labels_anchor = anchor_labels[0]
            labels_pred = pred_labels[0]
            ws_dist = wasserstein_distance(labels_anchor, labels_pred)
            ws_dists.append(ws_dist)
            
        for count in range(5):
            # remove the 5 neigborhood predictions with the largest ws distances
            max_idx = ws_dists.index(max(ws_dists))
            ws_dists.pop(max_idx)
            indices_list.pop(max_idx)
        
        indices = indices_list    
        true_positive = 0
        for j in range(len(indices)):
            if indices[j] in true_neighbors:
                true_positive += 1
        recall = true_positive/num_true_neighbors
        precision = true_positive/(num_pred_neighbors-5)            
        recalls.append(recall)
        precisions.append(precision)
        
    from statistics import mean            
    avg_recall = mean(recalls)
    avg_precision = mean(precisions)
    print("number of evaluated queries: ", len(precisions))
    print("completed")
    print("average recall: ", avg_recall)
    print("average precision: ", avg_precision)
    return avg_recall, avg_precision 

def get_recall(sess, ops, database_output, queries_output, sq):
    # queries_output: latent vectors of the completed submaps from 00
    # database_output: latent vectors of the original submaps from 00
    
    print(len(queries_output))
    database_nbrs = KDTree(database_output)

    recalls = []
    precisions = []

    num_evaluated=0
    for i in range(len(queries_output)): # the range is 4651
        true_neighbors= QUERY_SETS[sq][str(i)] # the json file, which defines the true neighbors of all submaps in sequence 00
        num_true_neighbors = len(true_neighbors)
        num_pred_neighbors = num_true_neighbors
        if(len(true_neighbors)==0):
            continue
        num_evaluated+=1
        distances, indices = database_nbrs.query(np.array([queries_output[i]]),k=num_pred_neighbors)
        true_positive = 0
        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                true_positive += 1
        recall = true_positive/num_true_neighbors
        precision = true_positive/num_pred_neighbors            
        recalls.append(recall)
        precisions.append(precision)
    from statistics import mean            
    avg_recall = mean(recalls)
    avg_precision = mean(precisions)
    print("number of evaluated queries: ", len(precisions))
    print("completed")
    print("average recall: ", avg_recall)
    print("average precision: ", avg_precision)
    return avg_recall, avg_precision 


if __name__ == "__main__":
    evaluate()
    
