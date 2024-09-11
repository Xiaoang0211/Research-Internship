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

#params
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--positives_per_query', type=int, default=4, help='Number of potential positives in each training tuple [default: 2]')
parser.add_argument('--negatives_per_query', type=int, default=12, help='Number of definite negatives in each training tuple [default: 20]')
parser.add_argument('--batch_num_queries', type=int, default=3, help='Batch Size during training [default: 1]')
parser.add_argument('--dimension', type=int, default=256)
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')

FLAGS = parser.parse_args()

BATCH_NUM_QUERIES = FLAGS.batch_num_queries
EVAL_BATCH_SIZE = 1
NUM_POINTS = 4096 # number of point, this must match the down-sampled points in the file 
POSITIVES_PER_QUERY= FLAGS.positives_per_query
NEGATIVES_PER_QUERY= FLAGS.negatives_per_query
GPU_INDEX = FLAGS.gpu
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

# loading the json file, which defines the positives for all submaps in SemanticKITTI
query_sets = "./KITTI_all/positive_sequence_D-3_T-0.json"
f = open(query_sets, "r")
QUERY_SETS = json.load(f)

# results folder for the evaluation metrics
RESULTS_FOLDER="pretrained_results_D-3_T-0/without_labels/"
if not os.path.exists(RESULTS_FOLDER): os.mkdir(RESULTS_FOLDER)

model_file= "model.ckpt"

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


def evaluate_precision_recall1():
    sequences = ["00", "02", "05", "06", "07", "08"] # the test sequences
    avg_precisions = []  
    avg_precisions_original = []  
    avg_recalls = []  
    avg_recalls_original = []  
    for sq in sequences:
        LOG_DIR = 'log_fold{}/'.format(sq) # model directory, depending on which test sequence we are going to evaluate
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

            # load the model
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
            # load the global descriptors of completed scenes
            feature_queries = np.load('/home/user/pointnetvlad/log_fold{}/feature_database/queries_4096_completed/{}_PV_{}.npy'.format(sq,sq,sq))
            feature_database = np.load('/home/user/pointnetvlad/log_fold{}/feature_database/database_4096_completed/{}_PV_{}.npy'.format(sq,sq,sq))
            # load the global descriptors of raw scans
            feature_queries_original = np.load('/home/user/pointnetvlad/log_fold{}/feature_database/queries_1.2_4096_original/{}_PV_{}.npy'.format(sq,sq,sq))
            feature_database_original = np.load('/home/user/pointnetvlad/log_fold{}/feature_database/database_1.2_4096_original/{}_PV_{}.npy'.format(sq,sq,sq))
            
            # average recall and precision of completed scene
            avg_recall, avg_precision = get_recall1(sess, ops, feature_database, feature_queries, sq)
            # average recall and precision of raw scans
            avg_recall_original, avg_precision_original = get_recall1(sess, ops, feature_database_original, feature_queries_original, sq)

            avg_precisions.append(avg_precision)
            avg_precisions_original.append(avg_precision_original)
            avg_recalls.append(avg_recall)
            avg_recalls_original.append(avg_recall_original)

    output_file= RESULTS_FOLDER +'recalls_and_precisions_gt4096.txt'
    with open(output_file, "w") as output:
        output.write("Average Precision:\n")
        output.write(str(sequences))
        output.write("\n")
        output.write("completed: ")
        output.write(str(avg_precisions))
        output.write("\n")
        output.write("original: ")
        output.write(str(avg_precisions_original))
        output.write("\n\n")
        output.write("Average Recalls:\n")
        output.write(str(sequences))
        output.write("\n")
        output.write("completed: ")
        output.write(str(avg_recalls))
        output.write("\n")
        output.write("original: ")
        output.write(str(avg_recalls_original))
    
def evaluate_precision_recall2():
    sequences = ["00", "02", "05", "06", "07", "08"]
    avg_precisions = []  
    avg_precisions_original = []  
    avg_recalls = []  
    avg_recalls_original = []  
    for num_predictions in range(1,26):
        for sq in sequences:
            LOG_DIR = 'log_fold{}/'.format(sq) # model directory, depending on which test sequence we are going to evaluate
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

                # load the model
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
                # load the global descriptors of completed scenes (predicted/ground truth)
                feature_queries = np.load('/home/user/pointnetvlad/log_fold{}/feature_database/queries_4096_completed/{}_PV_{}.npy'.format(sq,sq,sq))
                feature_database = np.load('/home/user/pointnetvlad/log_fold{}/feature_database/database_4096_completed/{}_PV_{}.npy'.format(sq,sq,sq))
                # load the global descriptors of raw scans
                feature_queries_original = np.load('/home/user/pointnetvlad/log_fold{}/feature_database/queries_1.2_4096_original/{}_PV_{}.npy'.format(sq,sq,sq))
                feature_database_original = np.load('/home/user/pointnetvlad/log_fold{}/feature_database/database_1.2_4096_original/{}_PV_{}.npy'.format(sq,sq,sq))
                
                # average recall and precision of completed scenes
                avg_recall, avg_precision = get_recall2(sess, ops, feature_database, feature_queries, num_predictions, sq)
                # average recall and precision of raw scans
                avg_recall_original, avg_precision_original = get_recall2(sess, ops, feature_database_original, feature_queries_original, sq)

                avg_precisions.append(avg_precision)
                avg_precisions_original.append(avg_precision_original)
                avg_recalls.append(avg_recall)
                avg_recalls_original.append(avg_recall_original)
        
    # save
    avg_precisions = np.array(avg_precisions)
    avg_precisions_original = np.array(avg_precisions_original)
    avg_recalls = np.array(avg_recalls)
    avg_recalls_original = np.array(avg_recalls_original)

    avg_precisions = avg_precisions.reshape((25,6))
    avg_precisions_original = avg_precisions_original.reshape((25,6))
    avg_recalls = avg_recalls.reshape((25,6))
    avg_recalls_original = avg_recalls_original.reshape((25,6))

    output_file_avg_precisions = RESULTS_FOLDER + "avg_precisions4096x4.txt"
    output_file_avg_precisions_original = RESULTS_FOLDER + "avg_precisions_original4096x4.txt"
    output_file_avg_recalls = RESULTS_FOLDER + "avg_recalls4096x4.txt"
    output_file_recalls_original = RESULTS_FOLDER + "avg_recalls_original4096x4.txt"


    np.savetxt(output_file_avg_precisions, avg_precisions)
    np.savetxt(output_file_avg_precisions_original, avg_precisions_original)
    np.savetxt(output_file_avg_recalls, avg_recalls)
    np.savetxt(output_file_recalls_original, avg_recalls_original)

            



def get_latent_vectors(sess, ops, sq_dir):
    train_file_idxs = []
    listDir(sq_dir, train_file_idxs)
    train_file_idxs.sort()
    is_training=False

    #print(len(train_file_idxs))
    batch_num= BATCH_NUM_QUERIES*(1+POSITIVES_PER_QUERY+NEGATIVES_PER_QUERY)
    q_output = []
    for q_index in tqdm(range(len(train_file_idxs)//batch_num)):
        file_names=train_file_idxs[q_index*batch_num:(q_index+1)*(batch_num)]
        queries=load_pc_files(file_names)
        # queries= np.expand_dims(queries,axis=1)
        q1=queries[0:BATCH_NUM_QUERIES]
        q1=np.expand_dims(q1,axis=1)
        #print(q1.shape)

        q2=queries[BATCH_NUM_QUERIES:BATCH_NUM_QUERIES*(POSITIVES_PER_QUERY+1)]
        q2=np.reshape(q2,(BATCH_NUM_QUERIES,POSITIVES_PER_QUERY,NUM_POINTS,3))

        q3=queries[BATCH_NUM_QUERIES*(POSITIVES_PER_QUERY+1):BATCH_NUM_QUERIES*(NEGATIVES_PER_QUERY+POSITIVES_PER_QUERY+1)]
        q3=np.reshape(q3,(BATCH_NUM_QUERIES,NEGATIVES_PER_QUERY,NUM_POINTS,3))
        feed_dict={ops['query']:q1, ops['positives']:q2, ops['negatives']:q3, ops['is_training_pl']:is_training}
        o1, o2, o3=sess.run([ops['q_vec'], ops['pos_vecs'], ops['neg_vecs']], feed_dict=feed_dict)
        
        o1=np.reshape(o1,(-1,o1.shape[-1]))
        o2=np.reshape(o2,(-1,o2.shape[-1]))
        o3=np.reshape(o3,(-1,o3.shape[-1]))

        out=np.vstack((o1,o2,o3))
        q_output.append(out)

    q_output=np.array(q_output)
    if(len(q_output)!=0):  
        q_output=q_output.reshape(-1,q_output.shape[-1])

    #handle edge case
    for q_index in tqdm(range((len(train_file_idxs)//batch_num*batch_num),len(train_file_idxs))):
        file_names=train_file_idxs[q_index]
        queries=load_pc_files([file_names])
        queries= np.expand_dims(queries,axis=1)
    
        fake_queries=np.zeros((BATCH_NUM_QUERIES-1,1,NUM_POINTS,3))
        fake_pos=np.zeros((BATCH_NUM_QUERIES,POSITIVES_PER_QUERY,NUM_POINTS,3))
        fake_neg=np.zeros((BATCH_NUM_QUERIES,NEGATIVES_PER_QUERY,NUM_POINTS,3))
        q=np.vstack((queries,fake_queries))
        feed_dict={ops['query']:q, ops['positives']:fake_pos, ops['negatives']:fake_neg, ops['is_training_pl']:is_training}
        output=sess.run(ops['q_vec'], feed_dict=feed_dict)
        output=output[0]
        output=np.squeeze(output)
        if (q_output.shape[0]!=0):
            q_output=np.vstack((q_output,output))
        else:
            q_output=output

    #q_output=np.array(q_output)
    #q_output=q_output.reshape(-1,q_output.shape[-1])
    print(len(train_file_idxs))
    assert q_output.shape[0] == len(train_file_idxs)
    return q_output

def get_recall1(sess, ops, database_output, queries_output, sq):
    # queries_output: latent vectors of the completed submaps from 00
    # database_output: latent vectors of the original submaps from 00
    
    print(len(queries_output))
    database_nbrs = KDTree(database_output)

    recalls = []
    precisions = []

    num_evaluated=0
    for i in range(len(queries_output)): # the range is 4651
        try:
            true_neighbors= QUERY_SETS[sq][str(i)] # the json file, which defines the true neighbors of all submaps in sequence 00
            num_true_neighbors = len(true_neighbors)
            # if num_pred_neighbors = num_true_neighbors, then it means we try to retrieve all positives of the anchor submap
            # num_pred_neighbors can also be changed to other integer values, e.g. 5 for evaluating @5 precision
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
        except:
            continue
    from statistics import mean            
    avg_recall = mean(recalls)
    avg_precision = mean(precisions)
    print("number of evaluated queries: ", len(precisions))
    print("completed")
    print("average recall: ", avg_recall)
    print("average precision: ", avg_precision)
    return avg_recall, avg_precision 


def get_recall2(sess, ops, database_output, queries_output, num_predictions, sq):
    # queries_output: latent vectors of the completed submaps from 00
    # database_output: latent vectors of the original submaps from 00
    
    print(len(queries_output))
    database_nbrs = KDTree(database_output)

    recalls = []
    precisions = []

    num_evaluated=0
    for i in range(len(queries_output)): # the range is 4651
        try:
            true_neighbors= QUERY_SETS[sq][str(i)] # the json file, which defines the true neighbors of all submaps in sequence 00
            num_true_neighbors = len(true_neighbors)
            num_pred_neighbors = num_predictions
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
        except:
            continue
    from statistics import mean            
    avg_recall = mean(recalls)
    avg_precision = mean(precisions)
    print("number of evaluated queries: ", len(precisions))
    print("completed")
    print("average recall: ", avg_recall)
    print("average precision: ", avg_precision)
    return avg_recall, avg_precision 


def visualize_recall_precision(sequence, dist_thresh=3, time_thresh=30):   
    # this method generate the PR-curve for one sequence
    import matplotlib.pyplot as plt
    precision_all_gt = np.loadtxt("pretrained_results_D-{}_T-{}/without_labels/gt_completed/avg_precisions.txt".format(dist_thresh, time_thresh))
    recall_all_gt = np.loadtxt("pretrained_results_D-{}_T-{}/without_labels/gt_completed/avg_recalls.txt".format(dist_thresh, time_thresh))
    precision_gt = precision_all_gt[:,sequence]
    recall_gt = recall_all_gt[:,sequence]
    
    precision_all = np.loadtxt("pretrained_results_D-{}_T-{}/without_labels/pred_completed/avg_precisions4096x4.txt".format(dist_thresh, time_thresh))
    recall_all = np.loadtxt("pretrained_results_D-{}_T-{}/without_labels/pred_completed/avg_recalls4096x4.txt".format(dist_thresh, time_thresh))
    precision = precision_all[:,sequence]
    recall = recall_all[:,sequence]
    
    precision_all_original = np.loadtxt("pretrained_results_D-{}_T-{}/without_labels/pred_completed/avg_precisions_original4096.txt".format(dist_thresh, time_thresh))
    recall_all_original = np.loadtxt("pretrained_results_D-{}_T-{}/without_labels/pred_completed/avg_recalls_original4096.txt".format(dist_thresh, time_thresh))
    precision_original = precision_all_original[:,sequence]
    recall_original = recall_all_original[:,sequence]
    if sequence == 0:
        sq = '00'
    elif sequence == 1:
        sq = '02'
    elif sequence == 2:
        sq = '05'
    elif sequence == 3:
        sq = '06'
    elif sequence == 4:
        sq = '07'
    elif sequence == 5:
        sq = '08'
    fig, ax = plt.subplots()
    ax.plot(recall_gt, precision_gt, color='lime', marker='o', linestyle='-', label='GT-PointNetVLAD')
    ax.plot(recall, precision, color='cyan', marker='o', linestyle='-', label='JS3C-PointNetVLAD')
    ax.plot(recall_original, precision_original, color='magenta', marker='x', linestyle='--', label='PointNetVLAD')
    #add axis labels to plot
    ax.set_title('Precision-Recall Curve of Sequence {}'.format(sq), fontsize = 15,  fontweight="bold")
    ax.set_ylabel('Precision', fontsize = 15)
    ax.set_xlabel('Recall', fontsize = 15)
    ax.legend(loc='lower left',prop={'size': 15})
    ax.grid(visible=True)
    fig.savefig('pretrained_results_D-{}_T-{}/without_labels/gt4096x4/pr_curve_sq{}.png'.format(dist_thresh,time_thresh,sq))

def max_f1_score(sequence, dist_thresh, time_thresh):
    # this method evaluate the max F1-Score for one sequence
    if sequence == 0:
        sq = '00'
    elif sequence == 1:
        sq = '02'
    elif sequence == 2:
        sq = '05'
    elif sequence == 3:
        sq = '06'
    elif sequence == 4:
        sq = '07'
    elif sequence == 5:
        sq = '08'
    precision_all = np.loadtxt("pretrained_results_D-{}_T-{}/without_labels/gt_completed/avg_precisions.txt".format(dist_thresh,time_thresh))
    recall_all = np.loadtxt("pretrained_results_D-{}_T-{}/without_labels/gt_completed/avg_recalls.txt".format(dist_thresh,time_thresh))
    precisions = precision_all[:,sequence]
    recalls = recall_all[:,sequence]
    
    precision_all_original = np.loadtxt("pretrained_results_D-{}_T-{}/without_labels/gt_completed/avg_precisions_original.txt".format(dist_thresh,time_thresh))
    recall_all_original = np.loadtxt("pretrained_results_D-{}_T-{}/without_labels/gt_completed/avg_recalls_original.txt".format(dist_thresh,time_thresh))
    precisions_original = precision_all_original[:,sequence]
    recalls_original = recall_all_original[:,sequence]
    f1_scores = []
    f1_scores_original = []
    
    for precision, recall in zip(precisions, recalls):
        f1 = 2*(precision*recall)/(precision+recall)
        f1_scores.append(f1)
    f1_max = max(f1_scores)
    
    for precision, recall in zip(precisions_original, recalls_original):
        f1 = 2*(precision*recall)/(precision+recall)
        f1_scores_original.append(f1)
        
    f1_max_original = max(f1_scores_original)
    
    output_file = 'pretrained_results_D-{}_T-{}/without_labels/gt4096x4/'.format(dist_thresh,time_thresh) + 'f1_max_{}.txt'.format(sq)
    with open(output_file, "w") as output:
            output.write("max f1 score of completed scene: ")
            output.write(str(f1_max))
            output.write("\n")
            output.write("max f1 score of original scene: ")
            output.write(str(f1_max_original))

if __name__ == "__main__":
    # section 1
    '''
    # write the precisions and recalls to a .txt file
    evaluate_precision_recall1()
    '''
    
    # section 2
    '''
    # generating the PR-curves and evaluation with F1-Scores
    evaluate_precision_recall2()
    D = 3
    T = 0
    visualize_recall_precision(0, D, T)
    visualize_recall_precision(1, D, T)
    visualize_recall_precision(2, D, T)
    visualize_recall_precision(3, D, T)
    visualize_recall_precision(4, D, T)
    visualize_recall_precision(5, D, T)
    
    max_f1_score(0, D, T)
    max_f1_score(1, D, T)
    max_f1_score(2, D, T)
    max_f1_score(3, D, T)
    max_f1_score(4, D, T)
    max_f1_score(5, D, T)
    '''
