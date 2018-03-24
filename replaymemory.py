#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 14:12:48 2018

@author: lihaoruo
"""
import numpy as np

class Replay:
    def __init__(self, memory_size, batch_size, dtype=np.float32):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.dtype = dtype

        self.states = None
        self.actions = np.empty(self.memory_size, dtype=np.uint8)
        self.rewards = np.empty(self.memory_size)
        self.next_states = None
        self.terminals = np.empty(self.memory_size, dtype=np.uint8)

        self.pos = 0
        self.full = False

    def feed(self, experience):
        state, action, reward, next_state, done = experience
        #print state
        if self.states is None:
            self.states = np.empty((self.memory_size, ) + state.shape, dtype=self.dtype)
            self.next_states = np.empty((self.memory_size, ) + state.shape, dtype=self.dtype)

        self.states[self.pos][:] = state
        self.actions[self.pos]   = action
        self.rewards[self.pos]   = reward
        self.next_states[self.pos][:] = next_state
        self.terminals[self.pos] = done
        #print self.pos
        #print (self.rewards)
        self.pos += 1
        if self.pos >= (self.memory_size):
            self.full = True
            self.pos = 0

    def sample(self):
        if self.full == True:
            upper_bound = self.memory_size-15
        else:
            upper_bound = self.pos
        
        #sampled_indices = 999.
        #while self.rewards[sampled_indices] != 0 and self.rewards[sampled_indices] != 1:
        sampled_indices_idx = np.random.randint(0, upper_bound, size=1)
        sampled_indices = []
        for i in range(self.batch_size):
            sampled_indices.append(sampled_indices_idx[0]+i)
        sampled_indices = np.stack(sampled_indices)
        #print 'dfs', sampled_indices
            
        #print 'dsfadsfadsfadsfadsfadsfafafadsafds', sampled_indices
        #for i in range(5):
        #    sampled_indices.append(indices+i)
        #print sampled_indices
        return [self.states[sampled_indices],
                self.actions[sampled_indices],
                self.rewards[sampled_indices],
                self.next_states[sampled_indices],
                self.terminals[sampled_indices]]

