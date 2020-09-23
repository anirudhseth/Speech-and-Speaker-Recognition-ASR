# DT2119_SpeechandSpeakerRecognition
## LAB 1 :
The objective is to experiment with different features commonly used for speech analysis and
recognition.
1. compute MFCC features step-by-step
1. examine features
1. evaluate correlation between feature
1. compare utterances with Dynamic Time Warping
1. illustrate the discriminative power of the features with respect to words
1. perform hierarchical clustering of utterances
1. train and analyze a Gaussian Mixture Model of the feature vectors.

## LAB 2 :
The overall task is to implement and test methods for isolated word recognition:
1. combine phonetic HMMs into word HMMs using a lexicon
1. implement the forward-backward algorithm,
1. use it compute the log likelihood of spoken utterances given a Gaussian HMM
1. perform isolated word recognition
1. implement the Viterbi algorithm, and use it to compute Viterbi path and likelihood
1. compare and comment Viterbi and Forward likelihoods
1. implement the Baum-Welch algorithm to update the parameters of the emission probability
distributions

## LAB 3 :

Train and test a phone recogniser based on digit speech material from the TIDIGIT database:
1. using predefined Gaussian-emission HMM phonetic models, create time aligned phonetic
transcriptions of the TIDIGITS database,
1. define appropriate DNN models for phoneme recognition using Keras,
1. train and evaluate the DNN models on a frame-by-frame recognition score,
1. repeat the training by varying model parameters and input features
