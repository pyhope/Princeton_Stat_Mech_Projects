import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator

rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = '16'
rcParams['font.sans-serif'] = 'Arial'
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm'] = 'Arial'
rcParams['mathtext.it'] = 'Arial:italic'
rcParams['mathtext.bf'] = 'Arial:bold'
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

g2_h = np.loadtxt('g2_0.45.dat', unpack=True)
ref_h = np.loadtxt('g2_ref_0.45.dat', unpack=True)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(g2_h[0], g2_h[1], label='my simulation ($\phi = 0.45$)', lw = 0.7)
ax.plot(ref_h[0],ref_h[1], label='literature ($\phi = 0.45$)', zorder=0, lw = 0.7)
error = ref_h[1] * 0.05
ax.fill_between(ref_h[0], ref_h[1] - error, ref_h[1] + error, alpha=0.7, facecolor='C1', label=r'5% range')
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

ax.set_xlabel('r')
ax.set_ylabel('g(r)')
plt.legend(fancybox=False, edgecolor='black', fontsize = 14)
plt.savefig('/Users/yihangpeng/Desktop/courses/Stat_Mech/final/plot/g2_0.45.pdf')
plt.show()

#2, 5, 9