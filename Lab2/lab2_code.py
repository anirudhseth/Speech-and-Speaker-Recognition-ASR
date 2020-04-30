import numpy as np
from lab2_proto import concatHMMs

data = np.load('lab2_data.npz', allow_pickle=True)['data']

phoneHMMs = np.load('lab2_models_onespkr.npz', allow_pickle=True)['phoneHMMs'].item()
# phoneHMMs = np.load('lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item()



wordHMMs = {}
wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])

