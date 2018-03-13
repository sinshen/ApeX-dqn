#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 23:45:19 2018

@author: lihaoruo
"""
import threading
import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
from atari_wrappers import wrap_deepmind
from time import sleep
from rm import Memory
import random

GLOBAL_STEP = 0
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

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class AC_Network():
    def __init__(self,sess, s_size,a_size,scope,trainer):
        with tf.variable_scope(scope):
            #Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
            self.imageIn = tf.reshape(self.inputs,shape=[-1,84,84,1])
            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                inputs=self.imageIn,num_outputs=16,
                kernel_size=[8,8],stride=[4,4],padding='VALID')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                inputs=self.conv1,num_outputs=32,
                kernel_size=[4,4],stride=[2,2],padding='VALID')
            hidden = slim.fully_connected(slim.flatten(self.conv2),256,activation_fn=tf.nn.elu)
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
            
            #Only the worker network need ops for loss functions and gradient updating.
            if scope == 'global':
                self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.reward = tf.placeholder(tf.float32, [None])

                y = self.reward + gamma * self.target_v
                readout_action = tf.reduce_sum(tf.multiply(self.out, self.actions_onehot), reduction_indices=1)
                self.loss = tf.reduce_mean(tf.square(y - readout_action))
                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
                
                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))
                
    def train(self,sess,gamma):
        while not coord.should_stop():
            tree_idx, batch_memory, ISweights = memory.sample(1)
            observations = batch_memory[0][0]
            actions = batch_memory[1][0]
            rewards = batch_memory[2][0]
            v1 = batch_memory[3][0]
            #print np.shape(observations)
            #print np.shape(actions)
            #print np.shape(rewards)
            #print np.shape(v1)
            actions = np.hstack(actions)
            rewards = np.hstack(rewards)
            #print actions
            #print rewards
            v1 = np.hstack(v1)
            #print v1
            feed_dict = {self.target_v:v1,
                         self.inputs:np.vstack(observations),
                         self.actions:actions,
                         self.reward:rewards}
            l,g_n,v_n,_ = sess.run([self.loss,
                                    self.grad_norms,
                                    self.var_norms,
                                    self.apply_grads],
                                    feed_dict=feed_dict)
        
            UPDATE_EVENT.clear()        # updating finished
            ROLLING_EVENT.set()         # set roll-out available
            #return l/len(rollout), g_n, v_n, v1

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
        self.local_AC = AC_Network(sess, s_size, a_size, self.name, None)
        self.update_local_ops = update_target_graph('global', self.name)
        self.env = env
        
    def work(self,gamma,sess,coord,saver):
        global GLOBAL_STEP
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        epsilon = 0.2
        print ("Starting worker " + str(self.number))
        best_mean_episode_reward = -float('inf')
        with sess.as_default(), sess.graph.as_default():                 
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_reward = 0
                episode_step_count = 0
                d = False
                buffer_s, buffer_a, buffer_r, buffer_v = [], [], [], []
                s = self.env.reset()
                s = process_frame(s)
                epsilon = epsilon * 0.995
                #idx = replay_buffer.store_frame(s)
                while not d:
                    if not ROLLING_EVENT.is_set():                  # while global PPO is updating
                        ROLLING_EVENT.wait()
                        buffer_s, buffer_a, buffer_r, buffer_v = [], [], [], []
                    GLOBAL_STEP += 1
                    #Take an action using probabilities from policy network output.
                    if random.random() > epsilon:
                        a_dist_list = sess.run([self.local_AC.out], feed_dict={self.local_AC.inputs:[s]})[0]
                        a_dist = a_dist_list[0]
                        a = np.argmax(a_dist)
                    else:
                        a = random.randint(0, 5)
                        
                    s1, r, d, _ = self.env.step(a)
                    
                    if d == False:
                        s1 = process_frame(s1)
                    else:
                        s1 = s
                        
                    buffer_s.append(s)
                    buffer_a.append(a)
                    buffer_r.append(r)
                    #memory.store((s, a, r, s1))
                    v1 = sess.run(self.local_AC.out, feed_dict={self.local_AC.inputs:[s1]})[0]
                    #print v1
                    v1 = np.max(v1)
                    
                    buffer_v.append(v1)
                    episode_reward = episode_reward + r
                    s = s1                    
                    total_steps += 1
                    episode_step_count += 1
                    
                    # from now on, it will do update
                    
                    if d != True and len(buffer_a) == 10:
                        bs = np.vstack(buffer_s)
                        ba = np.vstack(buffer_a)
                        br = np.vstack(buffer_r)
                        bv = np.vstack(buffer_v)
                        """
                        print np.shape(bs)
                        print np.shape(ba)
                        print np.shape(br)
                        print np.shape(bv)
                        """
                        memory.store((bs, ba, br, bv))
                        #QUEUE.put(np.hstack((bs, ba, br, bv)))
                        buffer_s, buffer_a, buffer_r, buffer_v = [], [], [], []
                        sess.run(self.update_local_ops)
                        
                        
                    if d == True:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                #self.episode_mean_values.append(np.mean(episode_values))
                    
                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if self.name == 'worker_0' and episode_count % 5 == 0:
                        print('\n episode: ', episode_count, 'global_step:', \
                              GLOBAL_STEP, 'mean episode reward: ', np.mean(self.episode_rewards[-5:]))
                    print v1
                    
                    if episode_count % 50 == 0 and self.name == 'worker_0':
                        saver.save(sess,self.model_path+'/last-'+str(episode_count)+'.cptk')
                        print ("Saved Model")
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
s_size = 7056 # Observations are greyscale frames of 84 * 84 * 1
load_model = True
epsilon = 0.2
model_path = './last'
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
num_workers = 4 # Set workers ot number of available CPU threads
saver = tf.train.Saver(max_to_keep=5)
Replay_memory = 40000

with tf.Session() as sess:
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()
    ROLLING_EVENT.set()
    
    GLOBAL_STEP = 0
    coord = tf.train.Coordinator()
    memory = Memory(capacity=Replay_memory)
    #sess.run(tf.global_variables_initializer())
    master_network = AC_Network(sess, s_size,a_size,'global',trainer)  # Generate global network
    workers = []
    for i in range(num_workers):
        env = get_env(task)
        workers.append(Worker(env,i,s_size,a_size,trainer,model_path,global_episodes))
    
    sess.run(tf.global_variables_initializer())
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)

    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(gamma,sess,coord,saver)
        t = threading.Thread(target=worker_work,args=())
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    worker_threads.append(threading.Thread(target=master_network.train(sess, gamma),args=()))
    worker_threads[-1].start()
    coord.join(worker_threads)

