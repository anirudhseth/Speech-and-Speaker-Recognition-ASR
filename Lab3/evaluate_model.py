# import Levenshtein
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import seaborn as sn
sn.set()
from utils import plot_confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix


def evaluate_by_frame_state_level(predictions_idx, true_idx, stateList, model_name):
    acc = accuracy_score(predictions_idx, true_idx)
    print('Frame-by-frame at the state level: ', acc * 100, '%')

    # plot confusion matrix
    cm = confusion_matrix(true_idx, predictions_idx)
    plot_confusion_matrix(cm, len(stateList), model_name, 'Confusion matrix for frame-by-frame at state level') 
    return acc


def evaluate_by_frame_phoneme_level(predictions_idx, true_idx, stateList, model_name):
    predictions_states = [stateList[int(state)].split('_')[0] for state in predictions_idx]
    true_states = [stateList[int(state)].split('_')[0] for state in true_idx]
    acc = accuracy_score(true_states, predictions_states)
    print('Frame-by-frame at the phoneme level: ', acc * 100, '%')

    stateList_merge = [state.split('_')[0] for state in stateList]
    stateList_merge = [k for k, g in itertools.groupby(stateList_merge)]

    # plot confusion matrix
    cm = confusion_matrix(true_states, predictions_states)
    plot_confusion_matrix(cm, len(stateList_merge), model_name, 'Confusion matrix for frame-by-frame at phoneme level') 
    return acc


def edit_distance_state_level(predictions_idx, true_idx, stateList):
    predictions_states = [stateList[int(state)] for state in predictions_idx]
    true_states = [stateList[int(state)] for state in true_idx]

    predictions_states_merged = [k for k, g in itertools.groupby(predictions_states)]
    true_states_merged = [k for k, g in itertools.groupby(true_states)]
    return Levenshtein.distance(''.join(true_states_merged),''.join(predictions_states_merged))


def edit_distance_phoneme_level(predictions_idx, true_idx, stateList):
    predictions_states = [stateList[int(state)].split('_')[0] for state in predictions_idx]
    true_states = [stateList[int(state)].split('_')[0] for state in true_idx]

    predictions_states_merged = [k for k, g in itertools.groupby(predictions_states)]
    true_states_merged = [k for k, g in itertools.groupby(true_states)]
    return Levenshtein.distance(''.join(true_states_merged),''.join(predictions_states_merged))




    
    