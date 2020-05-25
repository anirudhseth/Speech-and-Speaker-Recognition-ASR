import numpy as np
from lab2_tools import *

def recursiveAlpha(n, alpha, log_emlik, log_startprob, log_transmat):
    if n == 0:
        alpha[0] = np.log(log_startprob) + log_emlik[0]
        return alpha[0]
    else:
        alpha[n-1] = recursiveAlpha(n-1, alpha, log_emlik, log_startprob, log_transmat)
        N,M = np.shape(log_emlik)
        for j in range(0, M):
            alpha[n][j] = logsumexp(alpha[n-1] + np.log(log_transmat[:,j])) + log_emlik[n][j]
        return alpha[n]

def concatTwoHMMs(hmm1, hmm2):
    """ Concatenates 2 HMM models

    Args:
       hmm1, hmm2: two dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be different for each)

    Output
       dictionary with the same keys as the input but concatenated models:
          startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models
   
    Example:
       twoHMMs = concatHMMs(phoneHMMs['sil'], phoneHMMs['ow'])

    See also: the concatenating_hmms.pdf document in the lab package
"""
    hmmOut = {}
    M1,D1 =hmm1['transmat'].shape
    M2,D2 = hmm2['transmat'].shape
    hmmOut['name']=hmm1['name']+hmm2['name']
    hmmOut['startprob'] = hmm2['startprob'] * hmm1['startprob'][M1-1]
    hmmOut['startprob'] = np.concatenate((hmm1['startprob'][0:M1-1], hmmOut['startprob']))
    mul = np.reshape(hmm1['transmat'][0:-1, -1], (M1-1, 1)) @ np.reshape(hmm2['startprob'], (1, M2))
    hmmOut['transmat'] =  np.concatenate((hmm1['transmat'][0:-1, 0:-1], mul), axis=1)
    tmp = np.concatenate((np.zeros([M2,M1-1]), hmm2['transmat']), axis=1)
    hmmOut['transmat'] = np.concatenate((hmmOut['transmat'], tmp), axis=0)
    hmmOut['means'] = np.vstack((hmm1['means'],hmm2['means']))
    hmmOut['covars'] = np.vstack((hmm1['covars'],hmm2['covars']))
    return hmmOut

    
# this is already implemented, but based on concat2HMMs() above
def concatHMMs(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: dictionary of models indexed by model name. 
       hmmmodels[name] is a dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models:
         startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    concat = hmmmodels[namelist[0]]
    for idx in range(1,len(namelist)):
        # print(hmmmodels[namelist[idx]])
        concat = concatTwoHMMs(concat, hmmmodels[namelist[idx]])
    return concat


def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """
    N,_ = log_emlik.shape;
    ll = 0;
    for i in range(N):
        ll += logsumexp(log_emlik[i, :] + np.log(weights));
    return ll

def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """
    
    alpha = np.zeros(np.shape(log_emlik))
    N=len(alpha)
    # recursiveAlpha(N-1, alpha, log_emlik, log_startprob, log_transmat)
    alpha[0][:] = log_startprob.T + log_emlik[0]

    for n in range(1,len(alpha)):
        for i in range(alpha.shape[1]):
            alpha[n, i] = logsumexp(alpha[n - 1] + log_transmat[:,i]) + log_emlik[n,i]
    return alpha, logsumexp(alpha[N-1])
   


def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """
    N, M = log_emlik.shape
    backward_prob = np.zeros((N,M))

    backward_prob[N-1, :] = 0.0

    for i in range(N-2,-1,-1):
        for k in range(M):
            # probability of transitioning from k to state l * probability of emitting symbol at state l at ts i+1 * recursive backward probability
            backward_prob[i,k] = logsumexp(log_transmat[k,:] + log_emlik[i+1,:] + backward_prob[i+1,:])

    return backward_prob

    

def viterbi(log_emlik, log_startprob, log_transmat, forceFinalState=True):
    """ Computes viterbi log likelihood and Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j
        forceFinalState: if True, start backtracking from the final state in
                  the model, instead of the best state at the last time step

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """
    N, M = log_emlik.shape # (# timesteps, # states)
    B = np.zeros((N,M))
    V = np.zeros((N,M)) 

    # initialisation
    V[0,:] = log_startprob + log_emlik[0,:] 

    # induction
    for t in range(1,N):
        # vectorise
        x = np.tile(V[t-1,:],(M,1)) + log_transmat.T
        V[t,:] = np.max(x, axis=1) + log_emlik[t,:]
        B[t,:] = np.argmax(x, axis=1)

    # recover best path, looking for state sequence S that maximises P(S,X|emission probs)
    # TODO if forceFinalState
    end_state = np.argmax(V[N-1,:])  
        
    viterbi_path = [B[N-1,end_state]]
    viterbi_loglik = np.max(V[N-1,:])

    s_star = int(end_state)
    for t in range(N-2,-1,-1):
        s_star = int(B[t+1,s_star]) # optimal state at timestep t
        viterbi_path.append(s_star)

    assert len(viterbi_path) == N

    return viterbi_loglik, viterbi_path[::-1]
    

def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """

def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
    """ Update Gaussian parameters with diagonal covariance

    Args:
         X: NxD array of feature vectors
         log_gamma: NxM state posterior probabilities in log domain
         varianceFloor: minimum allowed variance scalar
    were N is the lenght of the observation sequence, D is the
    dimensionality of the feature vectors and M is the number of
    states in the model

    Outputs:
         means: MxD mean vectors for each state
         covars: MxD covariance (variance) vectors for each state
    """
