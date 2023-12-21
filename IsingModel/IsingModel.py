#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle

class IsingModel:
    def __init__(self, N = 16, K = 0.2, dim = 2):
        self.dim = dim
        self.K = K
        self.N_1 = N
        if self.dim == 2:
            self.N_2 = N
        else:
            self.N_2 = 1
        self.state = np.ones((self.N_1, self.N_2), dtype=int)
    
    def print_state(self):
        for i in self.state:
            for j in i:
                if j == 1:
                    print('■', end=' ')
                else:
                    print('□', end=' ')
            if self.dim == 2:
                print()
        print()
    
    def flip(self, i, j):
        self.state[i, j] = - self.state[i, j]
    
    def calc_spin_E(self, i, j): # apply periodic boundary condition
        spin = self.state[i, j]
        if i == self.N_1 - 1: right = self.state[0, j]
        else: right = self.state[i + 1, j]
        if i == 0: left = self.state[self.N_1 - 1, j]
        else: left = self.state[i - 1, j]
        if self.dim == 1:
            return -self.K * spin * (left + right)
        if j == 0: above = self.state[i, self.N_2 - 1]
        else: above = self.state[i, j - 1]   
        if j == self.N_2 - 1: below = self.state[i, 0]
        else: below = self.state[i, j + 1]   
        return -self.K * spin * (above + below + left + right)
    
    def calc_E(self):
        E = 0
        for i in range(self.N_1):
            for j in range(self.N_2):
                E += self.calc_spin_E(i, j)
        return E / (self.N_1 * self.N_2) / 2.0
       
    def calc_M(self):
        return float(np.sum(self.state) / (self.N_1 * self.N_2))

def sweep(model):
    num_flips = model.N_1 * model.N_2
    for step in range(num_flips): # Metropolis MC sampling
        i, j = np.random.randint(0, model.N_1), np.random.randint(0, model.N_2)
        E = model.calc_spin_E(i, j)
        model.flip(i, j)
        E_prime = model.calc_spin_E(i, j)
        if np.random.uniform(0, 1) > np.exp(-(E_prime - E)):
            model.flip(i, j)

def MonteCarlo(model, num_step, num_init, sample_freq):
    E_spl, M_spl = np.zeros(int((num_step - num_init) / sample_freq)), np.zeros(int((num_step - num_init) / sample_freq))
    num_spl, E_evo, M_evo = 0, [model.calc_E()], [model.calc_M()]
    for step in range(num_step):
        sweep(model)
        E_evo.append(model.calc_E())
        M_evo.append(model.calc_M())
        if step >= num_init:
            if ((step - num_init) % sample_freq) == 0:
                E_spl[num_spl] = model.calc_E()
                M_spl[num_spl] = model.calc_M()
                num_spl += 1
    return E_spl, M_spl, E_evo, M_evo # return samples and instantaneous E, M

# Simulation

# Nlist = [16, 32, 64]
# Klist = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

# Data = dict()
# for dim in [1, 2]:
#     Data[dim] = dict()
#     for N in Nlist:
#         Data[dim][N] = dict()
#         for K in Klist:
#             Data[dim][N][K] = dict()
#             model = IsingModel(N, K, dim)
#             E_spl, M_spl, E_evo, M_evo = MonteCarlo(model, 5000, 1000, 1)
#             Data[dim][N][K]['Es'] = E_spl
#             Data[dim][N][K]['Ms'] = M_spl
#             Data[dim][N][K]['Ee'] = E_evo
#             Data[dim][N][K]['Me'] = M_evo

'''
redo = [(1,64,0.6),(1,64,0.7),(2,16,0.2),(2,16,0.3),(2,16,0.4),(2,32,0.3),(2,32,0.4),(2,64,0.3),(2,64,0.4)]
Ndata = dict()
for i in redo:
    model = IsingModel(i[1], i[2], i[0])
    E_spl, M_spl, E_evo, M_evo = MonteCarlo(model, 5000, 1000, 1)
    Ndata[i] = dict()
    Ndata[i]['Es'] = E_spl
    Ndata[i]['Ms'] = M_spl
    Ndata[i]['Ee'] = E_evo
    Ndata[i]['Me'] = M_evo

with open('newdata.pkl', 'wb') as f:
    pickle.dump(Ndata, f) # Save data
'''
