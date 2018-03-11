#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 09:19:04 2018

@author: lihaoruo
"""
import threading
import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
import gym
from atari_wrappers import wrap_deepmind
from time import sleep
from rm import Memory

GLOBAL_STEP = 0
# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def process_frame(image):
    image = np.reshape(image,[np.prod(image.shape)]) / 255.0
    return image

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class AC_Network():
    def __init__(self,s_size,a_size,scope,trainer):
        with tf.variable_scope(scope):
            #Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
            self.imageIn = tf.reshape(self.inputs,shape=[-1,84,84,1])
            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                inputs=self.imageIn,num_outputs=32,
                kernel_size=[8,8],stride=[4,4],padding='VALID')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                inputs=self.conv1,num_outputs=64,
                kernel_size=[4,4],stride=[2,2],padding='VALID')
            self.conv3 = slim.conv2d(activation_fn=tf.nn.elu,
                inputs=self.conv2,num_outputs=64,
                kernel_size=[3,3],stride=[1,1],padding='VALID')
            hidden = slim.fully_connected(slim.flatten(self.conv2),512,activation_fn=tf.nn.elu)

            #Output layers for policy and value estimations
            self.policy = slim.fully_connected(hidden ,a_size,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None)
            self.value = slim.fully_connected(hidden, 1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)
            self.out = self.value + (self.policy - tf.reduce_mean(self.policy, axis=1, keep_dims=True))
            #self.out = tf.nn.softmax(abc)
            
            #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            else:
                self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                """
                self.readout_action = tf.reduce_sum(tf.multiply(self.out, self.actions_onehot))
                self.loss = tf.reduce_mean(tf.square(self.target_v - self.readout_action))
                """
                self.readout_action = tf.reduce_sum(tf.multiply(self.out, self.actions_onehot))
                max_act_t = tf.one_hot(tf.argmax(self.out, axis=1), depth=6, dtype=tf.float32)
                q_max_a_t = tf.reduce_sum(max_act_t*self.target_v, axis=1)
                for i in range(len(self.rewards_plus)):
                    pass
                #Get gradients from local network using local losses
                #self.local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.train_op = trainer.minimize(self.loss)
                
    def train(self,rollout,sess,gamma,lam,bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[0]
        actions = rollout[1]
        rewards = rollout[2]
        next_observations = rollout[3]


        feed_dict = {self.master_network.target_v:rewards,
            self.master_network.inputs:np.vstack(observations),
            self.master_network.actions:actions}

        l,_ = sess.run([self.master_network.loss,
            self.master_network.train_op],
            feed_dict=feed_dict)
        return l/len(rollout)
                
class Worker():
    def __init__(self,env,name,s_size,a_size,trainer,model_path,global_episodes):
        self.name = "worker_" + str(name)
        self.number = name        
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        #self.master_network = AC_Network(s_size,a_size,'global',trainer)
        self.local_AC = AC_Network(s_size, a_size, self.name, None)
        self.update_local_ops = update_target_graph('global', self.name)
        self.env = env

    def work(self,gamma,lam,sess,coord,saver):
        global GLOBAL_STEP
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print ("Starting worker " + str(self.number))
        best_mean_episode_reward = -float('inf')
        with sess.as_default(), sess.graph.as_default():                 
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_reward = 0
                episode_step_count = 0
                d = False
                
                s = self.env.reset()

                s = process_frame(s)
                while not d:
                    GLOBAL_STEP += 1
                    #Take an action using probabilities from policy network output.
                    a_dist = sess.run([self.local_AC.out], feed_dict={self.local_AC.inputs:[s]})[0]
                    a = np.random.choice(a_dist[0],p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    s1, r, d, _ = self.env.step(a)
                    if d == False:
                        s1 = process_frame(s1)
                    else:
                        s1 = s
                    
                    v1 = sess.run(self.local_AC.out, 
                                  feed_dict={self.local_AC.inputs:[s]})[0]
                    
                    episode_buffer.append([s,a,r,s1,d])
                    memory.store((s, a, r, s1, d))

                    episode_reward += r
                    s = s1                    
                    total_steps += 1
                    episode_step_count += 1
                    tree_idx, batch_memory, ISweights = memory.sample(len(episode_buffer))
                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == 10 and d != True:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        #v1 = sess.run(self.local_AC.out, 
                        #    feed_dict={self.local_AC.inputs:[s]})[0]
                        #v1 = np.max(v1)
                        #print v1
                        #l,g_n,v_n = self.train(episode_buffer,sess,gamma,lam,v1)
                        #tree_idx, batch_memory, ISweights = memory.sample(len(episode_buffer))
                        #l = self.train(batch_memory, sess, gamma, lam, v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if d == True:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                #self.episode_mean_values.append(np.mean(episode_values))
                
                # Update the network using the episode buffer at the end of the episode.
                #if len(episode_buffer) != 0:
                    #l,g_n,v_n = self.train(episode_buffer,sess,gamma,lam,0.0)
                    #l = self.train(batch_memory, sess, gamma, lam, 0.0)
                    
                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if self.name == 'worker_0' and episode_count % 5 == 0:
                        print('\n episode: ', episode_count, 'global_step:', \
                              GLOBAL_STEP, 'mean episode reward: ', np.mean(self.episode_rewards[-5:]))

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    if episode_count > 20 and best_mean_episode_reward < mean_reward:
                        best_mean_episode_reward = mean_reward

                if self.name == 'worker_0':
                    sess.run(self.increment)
                    #if episode_count%1==0:
                        #print('\r {} {}'.format(episode_count, episode_reward),end=' ')
                episode_count += 1

def get_env(task):
    env_id = task.env_id
    env = gym.make(env_id)
    env = wrap_deepmind(env)
    return env

gamma = .99 # discount rate for advantage estimation and reward discounting
lam = 0.97 # GAE discount factor
s_size = 7056 # Observations are greyscale frames of 84 * 84 * 1
load_model = False
model_path = './model'
 # Get Atari games.
benchmark = gym.benchmark_spec('Atari40M')
# Change the index to select a different game.
task = benchmark.tasks[3]

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)
    
env = get_env(task)
a_size = env.action_space.n

global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
master_network = AC_Network(s_size,a_size,'global',trainer) # Generate global network
num_workers = 2 # Set workers ot number of available CPU threads
workers = []
memory = Memory(capacity=500000)
# Create worker classes
for i in range(num_workers):
    env = get_env(task)
    workers.append(Worker(env,i,s_size,a_size,trainer,model_path,global_episodes))
saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
        
    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(gamma,lam,sess,coord,saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    worker_threads.append(threading.Thread(target=master_network.train, ))
    coord.join(worker_threads)
    