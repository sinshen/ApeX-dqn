#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 00:01:43 2018

@author: lihaoruo
"""

import threading
import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
import random
from atari_wrappers import wrap_deepmind
from time import sleep

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
                inputs=self.imageIn,num_outputs=16,
                kernel_size=[8,8],stride=[4,4],padding='VALID')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                inputs=self.conv1,num_outputs=32,
                kernel_size=[4,4],stride=[2,2],padding='VALID')
            """
            self.conv3 = slim.conv2d(activation_fn=tf.nn.elu,
                inputs=self.conv2,num_outputs=64,
                kernel_size=[3,3],stride=[1,1],padding='VALID')
            """
            hidden = slim.fully_connected(slim.flatten(self.conv2),256,activation_fn=tf.nn.elu)

            #Output layers for policy and value estimations
            self.policy = slim.fully_connected(hidden ,a_size,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None)
            self.out = self.policy
            """
            self.value = slim.fully_connected(hidden, 1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)
            self.out = self.value + (self.policy - tf.reduce_mean(self.policy, axis=1, keep_dims=True))
            """
            #self.out = tf.nn.softmax(self.out)
            #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.reward = tf.placeholder(shape=[None], dtype=tf.float32)
                #rewardlist = self.reward
                
                self.readout_action = tf.reduce_sum(tf.multiply(self.out, self.actions_onehot))
                self.loss = tf.reduce_mean(tf.square(self.target_v - self.readout_action))
                """
                q_act_t = tf.reduce_sum(self.actions_onehot*self.out)
                max_act_t = tf.one_hot(tf.argmax(self.out, axis=1), depth=a_size, dtype=tf.float32)
                q_max_act_t = tf.reduce_sum(max_act_t*self.target_v)
                y = []
                for i in range(10):
                    q_max_act_t = rewardlist[9-i] + q_max_act_t * gamma
                    y.append(q_max_act_t)
                y.reverse()
                self.loss = tf.nn.l2_loss(tf.subtract(y, q_act_t))
                """
                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
                
                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))

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
        self.local_AC = AC_Network(s_size, a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)
        self.env = env
        
    def train(self,rollout,sess,gamma,bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions = rollout[:,1]
        rewards = rollout[:,2]
        #next_observations = rollout[:,3]
        #print bootstrap_value
        targetv = []
        for i in range(0, 10):
            bootstrap_value[i] = rewards[i] + gamma * np.max(bootstrap_value[i])
            targetv.append(bootstrap_value[i])
        #print np.shape(targetv)
        #print targetv
        #self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        #discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        
        feed_dict = {self.local_AC.target_v:targetv[0],
            self.local_AC.inputs:np.vstack(observations),
            self.local_AC.actions:actions,
            self.local_AC.reward:rewards}
        l,g_n,v_n,_ = sess.run([self.local_AC.loss,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)
        
        return l/len(rollout), g_n,v_n
        
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
                episode_buffer = []
                #episode_values = []
                episode_reward = 0
                episode_step_count = 0
                d = False
                
                s = self.env.reset()
                s = process_frame(s)
                epsilon = epsilon * 0.99
                while not d:
                    GLOBAL_STEP += 1
                    #Take an action using probabilities from policy network output.
                    if random.random() > epsilon:
                        a_dist_list = sess.run([self.local_AC.out], feed_dict={self.local_AC.inputs:[s]})[0]
                        a_dist = a_dist_list[0]
                        #print a_dist[0]
                        #a = np.random.choice(a_dist[0],p=a_dist[0])
                        #print a
                        #a = np.argmax(a_dist == a)
                        a = np.argmax(a_dist)
                        #print a
                    else:
                        a = env.action_space.sample()
                    
                    s1, r, d, _ = self.env.step(a)
                    if d == False:
                        s1 = process_frame(s1)
                    else:
                        s1 = s
                        
                    episode_buffer.append([s,a,r,s1,d])
                    #episode_values.append(v[0,0])

                    episode_reward += r
                    s = s1                    
                    total_steps += 1
                    episode_step_count += 1
                    
                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == 10 and d != True:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        episode_buffer = np.array(episode_buffer)
                        buffer_s = episode_buffer[:, 0]
                        v1 = sess.run(self.local_AC.out, feed_dict={self.local_AC.inputs:np.vstack(buffer_s)})
                        
                        l,g_n,v_n = self.train(episode_buffer,sess,gamma,v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if d == True:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                #self.episode_mean_values.append(np.mean(episode_values))
                
                # Update the network using the episode buffer at the end of the episode.
                #if len(episode_buffer) != 0:
                #    l,g_n,v_n = self.train(episode_buffer,sess,gamma,0.0)
                    
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

#max_episode_length = 1#300
gamma = .99 # discount rate for advantage estimation and reward discounting
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
master_network = AC_Network(s_size,a_size,'global',None) # Generate global network
num_workers = 8 # Set workers ot number of available CPU threads
workers = []
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
        worker_work = lambda: worker.work(gamma,sess,coord,saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)
