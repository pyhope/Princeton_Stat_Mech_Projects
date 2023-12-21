#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

class configeration:
    def __init__(self, atom_pos=None, sys_size=None, step=0):
        self.atom_pos = atom_pos
        self.sys_size = sys_size
        self.step = step
    def atom_num(self):
        return len(self.atom_pos)
    def copy(self):
        return configeration(self.atom_pos.copy(), self.sys_size.copy(), self.step)

def initialize(packing_fraction, diameter): # generate initial configeration
    unit = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]) # a cubic unit cell (length L = 1) with 2 atoms
    config = configeration()

    supercell = [4, 4, 4]
    config.atom_pos = np.tile(unit, (np.prod(supercell),1))
    grid = [np.arange(supercell[i]) for i in range(3)]
    config.atom_pos += np.repeat(np.array(np.meshgrid(*grid)).T.reshape(-1,3), len(unit), axis=0)
    config.sys_size = np.array(supercell) * 1.0

    # Fix the number and radius of atom, change the packing fraction by resizing the system
    resize = (len(unit) * np.pi*diameter**3 / 6 / packing_fraction)**(1/3)
    config.sys_size *= resize
    config.atom_pos *= resize
    return config

def PDC(vec, sys_size): # periodic boundary conditions
    for i in range(len(vec)):
        if vec[i] < -0.5 * sys_size[i]:
            vec[i] += sys_size[i]
        if vec[i] > 0.5 * sys_size[i]:
            vec[i] -= sys_size[i]
    return vec

class MonteCarloFail(Exception):
    pass

def MonteCarlo(configeration, diameter=1):
    change = np.random.uniform(-0.5, 0.5, configeration.atom_pos.shape)
    for atom in range(configeration.atom_num()):
        new_pos = configeration.atom_pos[atom] + change[atom]
        try:
            for i in range(configeration.atom_num()):
                if i == atom:
                    continue
                distance = np.linalg.norm(PDC(configeration.atom_pos[i] - new_pos, configeration.sys_size))
                if distance <= diameter:
                    raise MonteCarloFail
            configeration.atom_pos[atom] = PDC(new_pos, configeration.sys_size)
        except MonteCarloFail:
            pass
    configeration.step += 1
    return configeration

def bin(config, bins):
    dis_set = config.atom_pos[:,None] - config.atom_pos[None,:]
    for i in range(3):
        dis_set[dis_set[:,:,i] < -0.5 * config.sys_size[i], i] += config.sys_size[i]
        dis_set[dis_set[:,:,i] >  0.5 * config.sys_size[i], i] -= config.sys_size[i]
    num = np.histogram(np.sqrt((dis_set**2.0).sum(axis=2))[np.triu_indices(config.atom_num(),1)], bins, density=False)
    return 2*num[0]

def calc_g2(traj, num_bins=200):
    sys_size = traj[0].sys_size
    bins = np.linspace(0, 0.5 * min(sys_size), num_bins + 1)
    num = np.zeros(num_bins)
    for config in traj:
        num += bin(config, bins)
    num *= 1/(len(traj))
    density = config.atom_num() / np.prod(sys_size)
    g2 = num / (config.atom_num() * density * 4/3 * np.pi * (bins[1:]**3 - bins[:-1]**3))
    r = 0.5*(bins[1:] + bins[:-1])
    return g2, r

# main
for packing_fraction in [0.45]:
    diameter = 1
    config = initialize(packing_fraction, diameter)

    traj = [config.copy()]
    for step in range(100): # Waiting for the system to reach equilibrium
            config = MonteCarlo(config)

    for step in range(500):
        config = MonteCarlo(config)
        traj.append(config.copy())

    g2, r = calc_g2(traj) # Calculate the radial distribution function

    np.savetxt('g2_'+str(packing_fraction)+'.dat', np.array([r,g2]).T) # Save data

    ref_h = np.loadtxt('g2_ref_0.45.dat', unpack=True)

    plt.plot(r,g2)
    plt.plot(ref_h[0],ref_h[1])

    plt.xlabel('r')
    plt.ylabel('g(r)')
    plt.show()