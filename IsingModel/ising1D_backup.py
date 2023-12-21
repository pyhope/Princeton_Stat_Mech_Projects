import numpy as np
from matplotlib import pyplot as plt
import time

class IsingModel:
    def __init__(self, N, K = None):
        self.K = K
        self.N = N
        state = np.ones(N)
        self.state = state
    
    def print_params(self):
        print("\t%d atoms" % (self.N))
        print("\tK = %f" % self.K)
    
    def flip_spin(self, i):
        self.state[i] = - self.state[i]
    
    def calculate_energy_of_spin(self, i):
        spin = self.state[i] 
        if i == self.N - 1: spin_right = self.state[0]
        else: spin_right = self.state[i + 1]
        if i == 0: spin_left = self.state[self.N - 1]
        else: spin_left = self.state[i - 1]
        return -self.K * spin * (spin_left + spin_right)
    
    def plot_lattice(self):
        plt.figure()
        imgplot = plt.imshow([self.state])
        imgplot.set_interpolation('none') 
        plt.xticks(range(self.N))
        plt.yticks(None)
        plt.show()
    
    def calculate_lattice_energy_per_spin(self):
        E = 0.0
        for i in range(self.N):
            E += self.calculate_energy_of_spin(i)
        # factor of two for overcounting neighboring interactions.
        return E / 2.0 / self.N
       
    def calculate_net_magnetization_per_spin(self):
        return float(np.sum(self.state) / self.N)

def sweep_lattice(model):
    num_flips = model.N
    n_accepted = 0
    for flip in range(num_flips):
        i = np.random.randint(0, high=model.N)
        E_old = model.calculate_energy_of_spin(i)
        model.flip_spin(i)
        E_new = model.calculate_energy_of_spin(i)
        if np.random.uniform(0, 1) > np.exp(-(E_new - E_old)):
            model.flip_spin(i)
        else:
            n_accepted += 1
    return float(n_accepted / num_flips)

def simulate(model, num_sweeps, num_burn_sweeps, sample_frequency):
    t0 = time.time()

    print("Simulating Ising model:")
    model.print_params()
    print("\t%d total sweeps, %d of them burn sweeps" % (num_sweeps, num_burn_sweeps))
    print("\t\tSampling every %d sweeps" % sample_frequency)

    energy_samples = np.zeros((int((num_sweeps - num_burn_sweeps) / sample_frequency), ))
    magnetization_samples = np.zeros((int((num_sweeps - num_burn_sweeps) / sample_frequency), ))
   
    n_samples = 0
    fraction_accepted_during_burn_sweeps = 0
    fraction_accepted_during_sampling_sweeps = 0
    E_evolution, M_evolution = [model.calculate_lattice_energy_per_spin()], [model.calculate_net_magnetization_per_spin()]

    for sweep in range(num_sweeps):
        fraction_accepted = sweep_lattice(model)
        E_evolution.append(model.calculate_lattice_energy_per_spin())
        M_evolution.append(model.calculate_net_magnetization_per_spin())
        if sweep >= num_burn_sweeps:
            fraction_accepted_during_sampling_sweeps += fraction_accepted
            if ((sweep - num_burn_sweeps) % sample_frequency) == 0:
                energy_samples[n_samples] = model.calculate_lattice_energy_per_spin()
                magnetization_samples[n_samples] = model.calculate_net_magnetization_per_spin()
                n_samples += 1
        else:
            fraction_accepted_during_burn_sweeps += fraction_accepted
        
    print("\t\tFraction proposals accepted during burn sweeps = %f" % (1.0 * fraction_accepted_during_burn_sweeps / num_burn_sweeps))
    print("\t\tFraction proposals accepted during sampling regime = %f" % (1.0 * fraction_accepted_during_sampling_sweeps / (num_sweeps - num_burn_sweeps)))
    
    print("\t<E> = %f +/- %f" % (np.mean(energy_samples), np.std(energy_samples) / np.sqrt(n_samples)))
    print("\t<m> = %f +/- %f" % (np.mean(magnetization_samples), np.std(magnetization_samples) / np.sqrt(n_samples)))
    
    print("\tSimulation finished. Took %s sec." % (time.time() - t0))
    assert((num_sweeps - num_burn_sweeps) / sample_frequency)
    
    return energy_samples, magnetization_samples, E_evolution, M_evolution

Nlist = [16, 32, 64]
Klist = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
model = IsingModel(64, 0.2)
energy_samples, magnetization_samples, E_evolution, M_evolution = simulate(model, 1000, 500, 1)
#model.plot_lattice()
plt.plot(list(range(len(E_evolution))), E_evolution)
plt.show()