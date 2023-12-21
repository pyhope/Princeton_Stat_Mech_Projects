#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt

class configeration:
    def __init__(self, atom_pos=[], sys_size=None, step=0):
        self.atom_pos = atom_pos
        self.sys_size = sys_size
        self.step = step
    def atom_num(self):
        return len(self.atom_pos)
    def copy(self):
        return configeration(self.atom_pos.copy(), self.sys_size.copy(), self.step)

def initialize(packing_fraction, diameter): 
    config = configeration()
    # initial configeration: a 7*7*7 supercell (N_atom = 686) of a cubic unit cell (N_atom = 2, L = 1, bcc)
    supercell = 4
    for i in range(supercell):
        for j in range(supercell):
            for k in range(supercell):
                config.atom_pos.append([i, j, k])
                config.atom_pos.append([i + 0.5, j + 0.5, k + 0.5])
    config.atom_pos = np.array(config.atom_pos, dtype=float)
    config.sys_size = np.tile(supercell, 3) * 1.0
    # Fix the number and radius of atom, change the packing fraction by resizing the system
    resize = (2 * np.pi*diameter**3 / 6 / packing_fraction)**(1/3)
    config.atom_pos *= resize
    config.sys_size *= resize
    return config

def PDC(vec, sys_size): # apply periodic boundary condition
    for i in range(len(vec)):
        if vec[i] < -0.5 * sys_size[i]:
            vec[i] += sys_size[i]
        if vec[i] > 0.5 * sys_size[i]:
            vec[i] -= sys_size[i]
    return vec

class SweepFail(Exception):
    pass

def Sweep(config, diameter=1):
    for atom in range(config.atom_num()):
        new_pos = config.atom_pos[atom] + np.random.uniform(-0.5, 0.5, config.atom_pos.shape)[atom] # Randomly change the position of each atom
        try:
            for i in range(config.atom_num()):
                if i == atom:
                    continue
                distance = np.linalg.norm(PDC(config.atom_pos[i] - new_pos, config.sys_size)) # Calculate the distance between each two spheres
                if distance <= diameter:
                    raise SweepFail  # If an overlap of hard spheres occurs, the Monte Carlo fails for this step
            config.atom_pos[atom] = PDC(new_pos, config.sys_size)
        except SweepFail:
            pass
    config.step += 1
    return config

def bin(config, bins): # bin the radial distance for each frame
    dis_set = config.atom_pos[:,None] - config.atom_pos[None,:] 
    for i in dis_set:
        for j in i:
            j = PDC(j, config.sys_size)
    num = np.histogram(np.sqrt((dis_set**2.0).sum(axis=2))[np.triu_indices(config.atom_num(),1)], bins)
    return 2 * num[0]

def calc_g2(traj, num_bins=200): # Calculate and normalize the radial distribution function
    bins = np.linspace(0, traj[0].sys_size[0]/2, num_bins + 1)
    r = (bins[1:] + bins[:-1]) / 2
    num = np.zeros(num_bins)
    for config in traj:
        num += bin(config, bins)
    g2 = num / len(traj) / (config.atom_num()**2 / np.prod(traj[0].sys_size) * 4/3 * np.pi * (bins[1:]**3 - bins[:-1]**3))
    return g2, r

# Simulation

for packing_fraction in [0.45]:
    diameter = 1
    config = initialize(packing_fraction, diameter)

    traj = [config.copy()]
    for step in range(100): # Waiting for the system to reach equilibrium
            config = Sweep(config)

    for step in range(500):
        config = Sweep(config)
        traj.append(config.copy())

    g2, r = calc_g2(traj)

    np.savetxt('g2_'+str(packing_fraction)+'.dat', np.array([r,g2]).T) # Save data
    ref_h = np.loadtxt('g2_ref_0.45.dat', unpack=True)

    plt.plot(r,g2)
    plt.plot(ref_h[0],ref_h[1])

    plt.xlabel('r')
    plt.ylabel('g(r)')
    plt.show()