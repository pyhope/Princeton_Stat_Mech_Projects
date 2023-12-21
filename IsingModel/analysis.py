import numpy as np
import pickle as pkl
import IsingModel

with open('result.pkl', 'rb') as f:
    Data = pkl.load(f)
with open('newdata.pkl', 'rb') as g:
    Ndata = pkl.load(g)

for i in [(1, 16, 0.5), (1, 16, 0.6), (1, 16, 0.7)]:
    Ndata[i] = dict()
    model = IsingModel.IsingModel(i[1], i[2], i[0])
    Ndata[i]['Es'], Ndata[i]['Ms'], Ndata[i]['Ee'], Ndata[i]['Me'] = IsingModel.MonteCarlo(model, 5000, 1000, 1)

for i in Ndata.keys():
    if i != (2, 32, 0.4):
        Data[i[0]][i[1]][i[2]] = Ndata[i]

Dimlist = [1, 2]
Nlist = [16, 32, 64]
Klist = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
Elemlist = ['Es', 'Ms', 'Ee', 'Me', 'te', 'tm', 'Eavg', 'Mavg', 'Eerr', 'Merr']

for d in Dimlist:
    for n in Nlist:
        for k in Klist:
            data = Data[d][n][k]
            avg_E_evo, avg_M_evo = np.mean(data['Ee'][100:500]), np.mean(data['Me'][100:500])
            data['Eavg'], data['Mavg'] = np.mean(data['Es']), np.mean(data['Ms'])
            for i in range(len(data['Ee'])):
                if i > 1 and data['Ee'][i] > avg_E_evo:
                    data['te'] = i
                    break
            for j in range(len(data['Me'])):
                if j > 1 and data['Me'][j] < avg_M_evo:
                    data['tm'] = j
                    break
            data['Eerr'] = np.std(data['Es']) / np.sqrt(len(data['Es']) / data['te'])
            data['Merr'] = np.std(data['Ms']) / np.sqrt(len(data['Ms']) / data['tm'])

with open('plotdata.pkl', 'wb') as f:
    pkl.dump(Data, f)