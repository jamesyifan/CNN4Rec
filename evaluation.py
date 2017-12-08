import sys
import numpy as np
import argparse
import tensorflow as tf
import utils as ut
import sklearn.metrics as metrics
from model import CNN4Rec


dataset = 'kuwo'

class Args():
    is_training = False
    n_epochs = 50
    batch_size = 1
    keep_prob = 1.0
    learning_rate = 0.001
    decay = 0.98
    decay_steps = 1e3
    grad_cap = 0
    checkpoint_dir = 'save/{}'.format(dataset)

def parseArgs():
    args = Args()
    parser = argparse.ArgumentParser(description='CNN4Rec args')
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--batch', default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--dr', default=0.98, type=float)
    parser.add_argument('--ds', default=500, type=int)
    parser.add_argument('--keep', default=1.0, type=float)
    parser.add_argument('--init_from', default=None, type=str)
    command_line = parser.parse_args()

    args.n_epochs = command_line.epoch
    args.batch_size = command_line.batch
    args.learning_rate = command_line.lr
    args.decay = command_line.dr
    args.decay_steps = command_line.ds
    args.keep_prob = command_line.keep
    
    args.checkpoint_dir += ('_batch' + str(command_line.batch))
    args.checkpoint_dir += ('_lr' + str(command_line.lr))
    args.checkpoint_dir += ('_dr' + str(command_line.dr))
    args.checkpoint_dir += ('_ds' + str(command_line.ds))
    args.checkpoint_dir += ('_p' + str(command_line.keep))

    args.init_from = command_line.init_from
    return args

def evaluate(args):
    label, train_num, valid_num, test_num = ut.load_data()
    n_classes = label.shape[1] - 1
    args.n_classes = n_classes
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    args.batch_size = 1
    img_batch, label_batch = ut.load_test(args.batch_size)
    recall, precision, f1 = 0.0, 0.0, 0.0
    with tf.Session(config=gpu_config) as sess:
        model = CNN4Rec(args)
        saver = tf.train.Saver(tf.global_variables()) 
        ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Restore model from {} successfully!'.format(args.checkpoint_dir))
        else:
            print('Restore model from {} failed!'.format(args.checkpoint_dir))
            return
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        num_batches = test_num / args.batch_size
        for k in xrange(num_batches):
            imgs, labels = sess.run([img_batch, label_batch])
            preds = model.predict_label(sess, imgs)
            preds[preds>0.5] = 1
            preds[preds<=0.5] = 0
            preds = preds.T
            preds = preds.astype(int)
            #print preds
            #print labels
            precision += metrics.precision_score(labels, preds, average='micro')
            recall += metrics.recall_score(labels, preds, average='micro')
            f1 += metrics.f1_score(labels, preds, average='micro')
        precision = precision / num_batches
        recall = recall / num_batches
        f1 = f1 / num_batches
        #print recall

    return recall, precision, f1

if __name__=='__main__':
    args = parseArgs()
    res = evaluate(args)
    print('lr: {}\tbatch_size: {}\tdecay_steps: {}\tdecay_rate: {}\tkeep_prob: {}'.format(args.learning_rate, args.batch_size, args.decay_steps, args.decay, args.keep_prob))
    print('Recall: {}\tPrecision: {}\tF1: {}'.format(res[0], res[1], res[2]))
    sys.stdout.flush()
            

            
