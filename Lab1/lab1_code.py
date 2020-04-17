import numpy as np
import lab1_proto
from lab1_proto import enframe,preemp,windowing,powerSpectrum,logMelSpectrum,cepstrum,dtw,mfcc
import matplotlib.pyplot as plt
from lab1_tools import lifter
import scipy
# load in data
example = np.load('lab1_example.npz', allow_pickle=True)['example'].item()

def customPlot(p,title,meshPlot):
    if (meshPlot):
        plt.pcolormesh(p)
    else:
        plt.plot(p)
    plt.title(title)
    plt.show()

    ''' Calculates number of samples in window and in window shift based on sampling rate and 
    frame rate, frame shift in ms'''
def calc_window_len_shift(sampling_rate, frame_rate=20, frame_shift=10):
    # frame_rate is window size in ms, frame_shift is window shift in ms
    window_size = (frame_rate / 1000) * sampling_rate
    window_shift = (frame_shift / 1000) * sampling_rate
    return int(window_size), int(window_shift)


sampling_rate=example['samplingrate']
# test enframe
window_size, window_shift = calc_window_len_shift(sampling_rate)
window_frames = enframe(example['samples'], window_size, window_shift)
# make sure is the same as example['frames'] 
print('Our example windowing matches theirs:', np.array_equal(example['frames'], window_frames))
plt.pcolormesh(window_frames)
plt.title(f'Color mesh for window of samples on example data, window size {window_size} (20ms), window_shift {window_shift} (10ms)')



customPlot(example['samples'],'samples:speech samples',False)

pre_emph=preemp(window_frames)
if(np.array_equal(example['preemph'], pre_emph)):
    customPlot(pre_emph,'preemph: preemphasis',True)
windowed=windowing(pre_emph)
if(np.allclose(example['windowed'],windowed,atol=1e-08)):
    customPlot(windowed,'windowed: hamming window',True)
fft_=powerSpectrum(windowed,512)
if(np.allclose(example['spec'],fft_,atol=1e-08)):
    customPlot(fft_,'spec:abs(FFT)^2',True)
logMel=logMelSpectrum(fft_,sampling_rate,512)
if(np.allclose(example['mspec'],logMel,atol=1e-08)):
    customPlot(logMel,'mspec:Mel Filterbank',True)
mfcc_=cepstrum(logMel,13)
if(np.allclose(example['mfcc'],mfcc_,atol=1e-08)):
    customPlot(mfcc_,'mfcc:MFCCs',True)
lmfcc_=lifter(mfcc_)
if(np.allclose(example['lmfcc'],lmfcc_,atol=1e-08)):
    customPlot(lmfcc_,'lmfcc:Liftered MFCCs',True)


from lab1_proto import mfcc,mspec
data = np.load('lab1_data.npz',allow_pickle=True)['data']
for i in range(data.shape[0]):
    samples=data[i]['samples']
    s=mfcc(samples)
    t=mspec(samples)
    if(i==0):
        data_mfcc=s
        data_mspec=t
    else:
        data_mfcc=np.append(data_mfcc,s,axis=0)  
        data_mspec=np.append(data_mspec,t,axis=0)

plt.pcolormesh(np.corrcoef(data_mfcc.T))   ## how corrcoef works ?
plt.pcolormesh(np.corrcoef(data_mspec.T))

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=4).fit(data_mfcc)
labels = gmm.predict(data_mfcc)

from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2).fit_transform(data_mfcc)
plt.scatter(X_embedded[:,0],X_embedded[:,1], c=labels, s=40, cmap='viridis')

for i in [16,17,38,39]:
    samples=data[i]['samples']
    s=mfcc(samples)
    if(i==0):
        data_gmm=s
    else:
        data_gmm=np.append(data_mfcc,s,axis=0)  
gmm = GaussianMixture(n_components=32).fit(data_gmm)
labels = gmm.predict(data_gmm)


sample_=data.shape[0]
result=np.zeros([sample_,sample_])
for i in range(sample_):
    mfcc1=mfcc(data[i]['samples'])
    for j in range(sample_):
        mfcc2=mfcc(data[j]['samples'])
        result[i,j]=dtw(mfcc1,mfcc2,scipy.spatial.distance.euclidean)
plt.plot(result)

from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(result, 'complete')
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z)