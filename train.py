import os, sys
import tensorflow as tf
import argparse
import numpy as np
import utils as ut
from model import CNN4Rec
train_data_path = 'train.tfrecords'
valid_data_path = 'valid.tfrecords'
dataset = 'kuwo'
error_during_training = False
class Args():
    is_training = True
    n_epochs = 10
    batch_size = 50
    keep_prob = 1.0
    learning_rate = 0.001
    decay = 0.98
    decay_steps = 1*1e2
    eval_point = 1*1e3
    grad_cap = 0
    checkpoint_dir = 'save/{}'.format(dataset)
def parseArgs():
    args = Args()
    parser = argparse.ArgumentParser(description='CNN4Rec args')
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--batch', default=50, type=int)
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

def train(args):
    label, train_num, valid_num, test_num = ut.load_data()
    n_classes = label.shape[1] - 1
    args.n_classes = n_classes
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    img_batch, label_batch = ut.load_train(args.batch_size, args.n_epochs)
    with tf.Session(config=gpu_config) as sess:
        #input_imgs = tf.placeholder(tf.int64, [self.args.batch_size, None], name='input')
        #input_labels = tf.placeholder(tf.int64, [self.args.bathc_size, None], name='output')
        #model = CNN4Rec(args, input_imgs, input_labels)
        model = CNN4Rec(args)
        if args.init_from is not None:
            ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                model.saver.restore(sess. ckpt.model_checkpoint_path)
                print 'Restore model from: {}'.foramt(args.checkpoint_dir)
        else:
            sess.run(tf.global_variables_initializer())
            print 'Randomly initialize model'
        valid_losses = []
        best_step = -1
        best_epoch = -1
        best_loss = 100.0
        error_during_train = False
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        num_batches = train_num / args.batch_size
        for epoch in xrange(args.n_epochs):
            epoch_cost = []
            for k in xrange(num_batches):
                imgs, labels = sess.run([img_batch, label_batch])
                fetches = [model.cost, model.global_step, model.lr, model.train_op]
                feed_dict = {model.imgs: imgs, model.labels: labels}
                cost, step, lr, _ =  sess.run(fetches, feed_dict) 
                epoch_cost.append(cost)
                if np.isnan(cost):
                    print(str(epoch) + ':Nan error!')
                    error_during_train = True
                    return
                if step == 1 or step % args.decay_steps == 0:
                    avgc = np.mean(epoch_cost)
                    print('Epoch {}\tProgress {}/{}\tlr: {}\tloss: {}'.format(epoch, k, num_batches, lr,  avgc))
                if step % args.eval_point == 0:
                    valid_loss = eval_validation(model, sess, valid_num)
                    valid_losses.append(valid_loss)
                    print('Evaluation loss after step {}: {}'.format(step, valid_loss))
                    if valid_loss < best_loss:
                        best_epoch = epoch
                        best_step = step
                        best_loss = valid_losses[-1]
                        ckpt_path = os.path.join(args.checkpoint_dir, 'model.ckpt')
                        model.saver.save(sess, ckpt_path, global_step=step)
                        print('model saved to {}'.format(ckpt_path))
                        sys.stdout.flush()
        coord.request_stop()
        coord.join(threads)
        print('Best evaluation loss appears in epoch {}, step {}. Lowest loss: {}'.format(best_epoch, best_step, best_loss))
        return
                    
def eval_validation(model, sess, valid_num):
    valid_batches = valid_num/args.batch_size
    valid_loss = []
    img_batch, label_batch = ut.load_valid(args.batch_size)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in xrange(valid_batches):
        imgs, labels = sess.run([img_batch, label_batch])
        fetches = model.cost
        feed_dict = {model.imgs: imgs, model.labels: labels}
        cost = sess.run(fetches, feed_dict)
        if np.isnan(cost):
            print('Evaluation loss Nan!')
            sys.exit(1)
        valid_loss.append(cost)
    return np.mean(valid_loss)

if __name__ == '__main__':
    args = parseArgs()
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    print('batch: {}\tepoch: {}\tkeep: {}'.format(args.batch_size, args.n_epochs, args.keep_prob))
    train(args)
    
