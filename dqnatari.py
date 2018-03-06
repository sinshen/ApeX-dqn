#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 23:02:14 2018

@author: lihaoruo
"""
import threading
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
from rm import Memory
from atari_wrappers import wrap_deepmind

# hyper-parameters
N_WORKER = 1
S_DIM = 7056
GAMMA = 0.99
EP_LEN = 200
MIN_BATCH_SIZE = 32
batch_size = 1
learning_rate = 1e-4
decay_rate = 0.99
EP_MAX = 1000.
Replay_memory = 5000000
benchmark = gym.benchmark_spec('Atari40M')
task = benchmark.tasks[6]
#np.set_printoptions(threshold=np.nan)

def get_env(task):
    env_id = task.env_id
    env = gym.make(env_id)
    env = wrap_deepmind(env)
    return env

env = get_env(task)

def process_frame(image):
    image = np.reshape(image,[np.prod(image.shape)]) / 255.0
    return image

class ApeX(object):
    def __init__(self):
        self.sess = tf.Session()
        self.t_state = tf.placeholder(tf.float32, shape=[None, S_DIM])
        self.t_states = tf.reshape(self.t_state, [-1, 84, 84, 1])
        
        self.probs = self.atari_model(img_in=self.t_states, num_actions=6, scope='net')
        self.sampling_prob = tf.nn.softmax(self.probs)
        
        self.t_actions = tf.placeholder(tf.int32, shape=[None])
        self.action_onehot = tf.one_hot(self.t_actions, 6, dtype=tf.float32)
        self.y = tf.placeholder(tf.float32, shape=[None])

        out_action = tf.reduce_sum(tf.multiply(self.sampling_prob, self.action_onehot), reduction_indices=1)
        loss = tf.reduce_mean(tf.square(self.y - out_action))
        self.train_op = tf.train.RMSPropOptimizer(learning_rate, decay_rate).minimize(loss)
        self.sess.run(tf.global_variables_initializer())
        
    def atari_model(self, img_in, num_actions, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            out = img_in
            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu, inputs=out, num_outputs=16, kernel_size=[8,8],stride=[4,4],padding='VALID')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu, inputs=self.conv1, num_outputs=32, kernel_size=[4,4],stride=[2,2],padding='VALID')
            hidden = slim.fully_connected(slim.flatten(self.conv2),256,activation_fn=tf.nn.elu)
            with tf.variable_scope("action_value"):
                action_out = slim.fully_connected(hidden, num_outputs=512, activation_fn=tf.nn.relu)
                action_out = slim.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)
            with tf.variable_scope("state_value"):
                state_out = slim.fully_connected(hidden, num_outputs=512, activation_fn=tf.nn.relu)
                state_out = slim.fully_connected(state_out, num_outputs=1, activation_fn=None)
            with tf.variable_scope("Q_value"):
                Q_out = state_out + (action_out - tf.reduce_mean(action_out, axis=1, keep_dims=True))
            return Q_out

    def update(self):
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()                     # wait until get batch of data
                #self.sess.run(self.update_oldpi_op)     # copy pi to old pi
                """
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]      # collect data from all workers
                data = np.vstack(data)
                s, a, r = data[:, :S_DIM], data[:, S_DIM: S_DIM + A_DIM], data[:, -1:]
                """
                tree_idx, batch_memory, ISweights = memory.sample(MIN_BATCH_SIZE)
                s, a, r, s_ = batch_memory[0], batch_memory[1], batch_memory[2], batch_memory[3]
                s = np.reshape(s, [-1, 7056])
                s_ = np.reshape(s_, [-1, 7056])
                a = np.reshape(a, [-1])
                r = np.reshape(r, [-1,])
                y_batch = []
                
                print 'a', np.shape(s)
                print np.shape(a)
                print np.shape(r)
                print np.shape(s_)
                
                q_out1 = self.sess.run(self.sampling_prob, feed_dict={self.t_state:s_})
                print 'b', np.shape(q_out1)
                for i in range(0, 32):
                    y_batch.append(r[i] + GAMMA * np.max(q_out1[i]))
                """
                print np.shape(y_batch)
                """
                self.sess.run(self.train_op, feed_dict={self.t_state: s, self.t_actions: a, self.y: y_batch})
                #[self.sess.run(self.train_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(UPDATE_STEP)]
                UPDATE_EVENT.clear()        # updating finished
                GLOBAL_UPDATE_COUNTER = 0   # reset counter
                ROLLING_EVENT.set()         # set roll-out available
    
    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sampling_prob, {self.t_state: s})[0]
        return np.clip(a, -2, 2)
    
    
class Worker(object):
    def __init__(self, env, name):
        self.name = name
        self.env = env
        self.apex = GLOBAL_APEX
        
    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            s = self.env.reset()
            s = process_frame(s)
            ep_r = 0
            for t in range(EP_LEN):
                if not ROLLING_EVENT.is_set():                  # while global PPO is updating
                    ROLLING_EVENT.wait()                        # wait until PPO is updated
    
                a_prob = self.apex.choose_action(s)
                a = np.argmax(a_prob)
                
                s_, r, done, _ = self.env.step(a)
                s_ = process_frame(s_)
                
                GLOBAL_UPDATE_COUNTER += 1               # count to minimum batch size, no need to wait other workers
                if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:

                    memory.store((s, a, r, s_))
                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()       # stop collecting data
                        UPDATE_EVENT.set()          # globalPPO update

                    if GLOBAL_EP >= EP_MAX:         # stop training
                        COORD.request_stop()
                        break
                s = s_
                ep_r += r
                
if __name__ == '__main__':
    GLOBAL_APEX = ApeX()
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()            # not update now
    ROLLING_EVENT.set()             # start to roll out
    workers = [Worker(env=env, name=i) for i in range(N_WORKER)]
    
    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    GLOBAL_RUNNING_R = []
    COORD = tf.train.Coordinator()
    memory = Memory(capacity=Replay_memory)
    threads = []
    for worker in workers:          # worker threads
        t = threading.Thread(target=worker.work, args=())
        t.start()                   # training
        threads.append(t)
    # add a PPO updating thread
    threads.append(threading.Thread(target=GLOBAL_APEX.update,))
    threads[-1].start()
    COORD.join(threads)

