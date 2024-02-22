import torch
import numpy as np
import librosa
from .sampling import *

EPS = np.finfo(float).eps

''' General audio operations '''

def energy(x, frame_length=2048):
    '''
    x: batch x # channels x # timesteps
    Returns: en: (batch, ) peak energey of each sample
    '''
    ws = windowing(x, frame_length=frame_length)
    en = np.mean(ws**2)
    return en

def peak_energy(x, frame_length=2048):
    '''
    x: batch x # channels x # timesteps
    Returns: en: (batch, ) peak energey of each sample
    '''
    ws = windowing(x, frame_length=frame_length)
    en = np.max(np.mean(ws**2, axis=0))
    return en

def peak_perc_energy(x, frame_length=2048, perc=0.2):
    '''
    x: batch x # channels x # timesteps
    Returns: en: (batch, ) average 20% peak energey of each sample
    '''
    ws = windowing(x, frame_length=frame_length)
    en = np.mean(ws**2, axis=0)
    k = int(perc * ws.shape[-1])
    sel_en = np.mean(np.partition(en, -k, axis=-1)[-k:])
    return sel_en

def activity_threshold(en, db_thresh=-60):
    '''
    en: energy scalar value
    Returns: mask: batch x # frames, whether above the db threshold or not
    '''
    return (10.0 * np.log10(en + EPS)) > db_thresh

def activity_detect(ws, db_thresh=-60):
    '''
    ws: batch x # features x # frames
    Returns: mask: batch x # frames, whether above the db threshold or not
    '''
    ws_ms = np.mean(ws**2, axis=0)
    mask = (10.0 * np.log10(ws_ms + EPS)) > db_thresh
    return mask

def active_energy(x, frame_length=2048, db_thresh=-60):
    '''
    x: # timesteps
    Returns: en: average energey of active part of each sample
    '''
    ws = windowing(x, frame_length=frame_length)

    en = np.mean(ws ** 2.0, axis=0)
    mask = (10.0 * np.log10(en + EPS)) > db_thresh
    active_en = en[mask]

    if len(active_en) > 0:
        en = np.mean(active_en)
    else:
        en = 0
    return en

def windowing(x, frame_length=2048):
    # Returns: (frame length, # frames)
    return librosa.util.frame(x, frame_length=frame_length, hop_length=frame_length)

def generate_gaussian_noise(length):
    return np.random.normal(size=length)

def volume_normalize(x, target_level=-25, frame_length=2048):
    '''Normalize the signal to the target volume level'''
    en = energy(x, frame_length=2048)
    
    rms = np.sqrt(np.mean(x ** 2.0))    
    scalar = 10.0 ** (target_level / 20.0) / (rms + EPS)
    x = x * scalar
    return x, scalar

def active_volume_normalize(x, target_level=-25, db_thresh=-60, frame_length=2048, ):
    '''Normalize the signal to the target volume level'''
    en = active_energy(x, frame_length=frame_length, db_thresh=db_thresh)
    if en > 0:
        rms = np.sqrt(en)
        scalar = 10.0 ** (target_level / 20.0) / (rms + EPS)
        x = x * scalar
    else:
        scalar = 1.0
    return x, scalar

def peak_volume_normalize(x, target_level=-25, frame_length=2048):
    '''Normalize the signal to the target volume level'''
    en = peak_energy(x, frame_length=frame_length)
    rms = np.sqrt(en)
    scalar = 10.0 ** (target_level / 20.0) / (rms + EPS)
    x = x * scalar
    return x, scalar

def peak_perc_volume_normalize(x, target_level=-25, frame_length=2048, perc=0.2):
    '''Normalize the signal to the target volume level'''
    en = peak_perc_energy(x, frame_length=frame_length, perc=perc)
    rms = np.sqrt(en)
    scalar = 10.0 ** (target_level / 20.0) / (rms + EPS)
    x = x * scalar
    return x, scalar

def fix_clipped(audio, clipping_threshold=0.99):
    scalar = 1.0

    maxamplevel = np.max(np.abs(audio))
    if maxamplevel > clipping_threshold:
        maxamplevel = maxamplevel / clipping_threshold
        scalar = 1.0 / maxamplevel

        audio = audio * scalar
    
    return audio, scalar