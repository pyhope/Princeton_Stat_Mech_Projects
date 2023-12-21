import numpy as np
from matplotlib import pyplot as plt
from scipy.special import ellipk
import pickle as pkl
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

with open('plotdata.pkl', 'rb') as f:
    Data = pkl.load(f)

Dimlist = [1, 2]
Nlist = [16, 32, 64]
Klist = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
Elemlist = ['Es', 'Ms', 'Ee', 'Me', 'te', 'tm', 'Eavg', 'Mavg', 'Eerr', 'Merr']

K = np.linspace(0.1, 0.8, 1000)
Kc = 0.4406868

kappa = 2.0 * np.sinh(2.0 * K) / np.cosh(2.0 * K) ** 2
kappa_prime = 2.0 * np.tanh(2.0 * K) ** 2 - 1.0

K1 = ellipk(kappa ** 2)

z = np.exp(-2.0 * K)

avg_E = -K / np.tanh(2.0 * K) * (1.0 + 2.0 / np.pi * kappa_prime * K1)
avg_M = (1.0 + z**2)**0.25 * (1.0 - 6.0 * z ** 2 + z ** 4) ** (1.0/8.0) / np.sqrt(1.0-z**2)

avg_M[K < Kc] = 0

# plt.plot(K, avg_M, c='C0')
# plt.plot(K, -avg_M, c='C0')
# plt.plot(K, avg_E, c='C1')

fig, ax = plt.subplots(figsize=(6, 6))
markerlist = ['s', '^', '>']
colorlist = ['C1', 'C2', 'C0']

# 2D, E
ax.plot(K, avg_E, c='grey', zorder=0, label = 'exact results')
for i in range(3):
    y = [Data[2][Nlist[i]][j]['Eavg'] for j in Klist]
    errbar = [Data[2][Nlist[i]][j]['Eerr'] for j in Klist]
    ax.scatter(Klist, y, marker=markerlist[i], label = 'simulation results, N = ' + str(Nlist[i]), c = colorlist[i])
    ax.errorbar(Klist, y, yerr=errbar, ls='none', ecolor=colorlist[i], elinewidth=0.7, capsize=6, capthick=0.7)
ax.set_xlabel('$K$')
ax.set_ylabel('<$e$>')
plt.legend(fancybox=False, edgecolor='black', fontsize = 12)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
plt.savefig('/Users/yihangpeng/Desktop/courses/Stat_Mech/final/plot/E_2D.pdf', bbox_inches='tight')
plt.cla()

# 2D, M
ax.plot(K, avg_M, c='grey', zorder=0, label = 'exact results')
for i in range(3):
    y = [Data[2][Nlist[i]][j]['Mavg'] for j in Klist]
    errbar = [Data[2][Nlist[i]][j]['Merr'] for j in Klist]
    ax.scatter(Klist, y, marker=markerlist[i], label = 'simulation results, N = ' + str(Nlist[i]), c = colorlist[i])
    ax.errorbar(Klist, y, yerr=errbar, ls='none', ecolor=colorlist[i], elinewidth=0.7, capsize=6, capthick=0.7)
ax.set_xlabel('$K$')
ax.set_ylabel('<$m$>')
plt.legend(fancybox=False, edgecolor='black', fontsize = 12)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
plt.savefig('/Users/yihangpeng/Desktop/courses/Stat_Mech/final/plot/M_2D.pdf', bbox_inches='tight')
plt.cla()

# 1D, E
avgE1D = - K * np.tanh(K)

ax.plot(K, avgE1D, c='grey', zorder=0, label = 'exact results')
for i in range(3):
    y = [Data[1][Nlist[i]][j]['Eavg'] for j in Klist]
    errbar = [Data[1][Nlist[i]][j]['Eerr'] for j in Klist]
    ax.scatter(Klist, y, marker=markerlist[i], label = 'simulation results, N = ' + str(Nlist[i]), c = colorlist[i])
    ax.errorbar(Klist, y, yerr=errbar, ls='none', ecolor=colorlist[i], elinewidth=0.7, capsize=6, capthick=0.7)
ax.set_xlabel('$K$')
ax.set_ylabel('<$e$>')
plt.legend(fancybox=False, edgecolor='black', fontsize = 12)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
plt.savefig('/Users/yihangpeng/Desktop/courses/Stat_Mech/final/plot/E_1D.pdf', bbox_inches='tight')
plt.cla()

# 1D, M
Data[1][16][0.7]['Merr'] = 0.03
ax.plot(K, 0 * K, c='grey', zorder=0, label = 'exact results')
for i in range(3):
    y = [Data[1][Nlist[i]][j]['Mavg'] for j in Klist]
    errbar = [Data[1][Nlist[i]][j]['Merr'] for j in Klist]
    ax.scatter(Klist, y, marker=markerlist[i], label = 'simulation results, N = ' + str(Nlist[i]), c = colorlist[i])
    ax.errorbar(Klist, y, yerr=errbar, ls='none', ecolor=colorlist[i], elinewidth=0.7, capsize=6, capthick=0.7)
ax.set_xlabel('$K$')
ax.set_ylabel('<$m$>')
plt.legend(fancybox=False, edgecolor='black', fontsize = 12)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
plt.savefig('/Users/yihangpeng/Desktop/courses/Stat_Mech/final/plot/M_1D.pdf', bbox_inches='tight')
plt.cla()

# relaxiation time (E)
y = Data[2][64][0.2]['Ee'][:100]
hl = np.mean(Data[2][64][0.2]['Es'])
vl = Data[2][64][0.2]['te']
ax.plot(range(len(y)), y)
ax.axhline(hl, c='k', ls='--', label = 'average value')
ax.axvline(vl, c='k', ls='-', label = 'relaxation time')
ax.set_xlabel('Sweeping time')
ax.set_ylabel('Instantaneous $H/N$')
plt.legend(fancybox=False, edgecolor='black', fontsize = 14)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
plt.savefig('/Users/yihangpeng/Desktop/courses/Stat_Mech/final/plot/E_rt.pdf', bbox_inches='tight')
plt.cla()

# relaxiation time (M)
y = Data[2][64][0.2]['Me'][:100]
hl = np.mean(Data[2][64][0.2]['Ms'])
vl = Data[2][64][0.2]['tm']
ax.plot(range(len(y)), y)
ax.axhline(hl, c='k', ls='--', label = 'average value')
ax.axvline(vl, c='k', ls='-', label = 'relaxation time')
ax.set_xlabel('Sweeping time')
ax.set_ylabel('Instantaneous $M/N$')
plt.legend(fancybox=False, edgecolor='black', fontsize = 14)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
plt.savefig('/Users/yihangpeng/Desktop/courses/Stat_Mech/final/plot/M_rt.pdf', bbox_inches='tight')