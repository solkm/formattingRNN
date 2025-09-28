#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 13:06:44 2022

@author: Sol
"""

from psychrnn.tasks.task import Task
import numpy as np
import math
import circledots_fxns as cf
rng = np.random.default_rng()

class DLPFC_combined(Task):
    def __init__(self, dt=10, tau=100, T=1200, N_batch=100, N_rec=100, in_noise=0.4, 
                 D_mstim_strength=0, MT_task=None, MT_simulator=None, MT_NdirOuts=72, 
                 saveMTactivity=False):
        N_in = 3 + MT_NdirOuts
        N_out = 72
        super().__init__(N_in, N_out, dt, tau, T, N_batch)
        self.N_rec = N_rec
        self.in_noise = in_noise
        self.D_mstim_strength = D_mstim_strength
        self.MTtask = MT_task
        self.MT_simulator = MT_simulator
        self.MT_NdirOuts = MT_NdirOuts
        self.saveMTactivity = saveMTactivity
        
    def generate_trial_params(self, batch, trial):
        
        """"Define parameters for each trial.
    
        Using a combination of randomness, presets, and task attributes, define the necessary trial parameters.
    
        Args:
            batch (int): The batch number that this trial is part of.
            trial (int): The trial number of the trial within the batch.
    
        Returns:
            dict: Dictionary of trial parameters.
    
        """
        # ----------------------------------
        # Define parameters of a trial
        # ----------------------------------
        params = dict()
        
        if trial==0: # new batch
            
            MT_inputs, _, _, self.MT_trial_params = self.MTtask.get_trial_batch()
            self.MT_output, MT_statevar = self.MT_simulator.run_trials(MT_inputs)
            
            if self.saveMTactivity==True:
                params['MTactivity'] = MT_statevar
        
        params['t_ring'] = self.MT_trial_params[trial]['t_ring']
        params['t_dots'] = self.MT_trial_params[trial]['t_dots']
        params['shown_deg'] = self.MT_trial_params[trial]['shown_deg']
        params['good_deg'] = self.MT_trial_params[trial]['good_deg']
        params['trial'] = trial
        
        shown_deg = params['shown_deg']
        good_deg = params['good_deg']
        
        R = np.zeros(72)
        if shown_deg is not None:
            for i in range(72):
                theta_deg = 5*i
                distFromShown_deg = cf.distance_deg(shown_deg, theta_deg)
    
                if distFromShown_deg <= 45: # accuracy threshold
                    A = 3.5 + 1.5 * np.cos(1.8 * math.radians(distFromShown_deg))
    
                    distFromGoodRew_deg = cf.distance_deg(good_deg, theta_deg)
                    B = 1 + 0.75 * np.cos(math.radians(distFromGoodRew_deg))
                        
                    R[i] = A * B
        
        params['reward'] = R

        return params

    def trial_function(self, time, params):
        """ Compute the trial properties at the given time.
    
        Based on the params compute the trial stimulus (x_t), correct output (y_t), and mask (mask_t) at the given time.
    
        Args:
            time (int): The time within the trial (0 <= time < T).
            params (dict): The trial params produced generate_trial_params()
    
        Returns:
            tuple:
    
            x_t (ndarray(dtype=float, shape=(N_in,))): Trial input at time given params.
            y_t (ndarray(dtype=float, shape=(N_out,))): Correct trial output at time given params.
            mask_t (ndarray(dtype=bool, shape=(N_out,))): True if the network should train to match the y_t, False if the network should ignore y_t when training.
        """

        # ----------------------------------
        # Initialize with noise
        # ----------------------------------
        x_t = np.sqrt(2 * self.alpha * self.in_noise**2) * rng.standard_normal(self.N_in)
        y_t = np.zeros(self.N_out)
        mask_t = np.zeros(self.N_out)
    
        # ----------------------------------
        # Retrieve parameters
        # ----------------------------------
        good_deg = params['good_deg']
        R = params['reward']
        trial = params['trial']
        ring_on = params['t_ring']
        dots_on = params['t_dots']
        # ----------------------------------
        # Compute values
        # ----------------------------------
        go = self.T - 100
        mask_t[:] = 0.2
        
        x_t[2:self.N_in-1] += self.MT_output[trial, int(time/self.dt), 2:self.N_in-1]
        
        if time > ring_on:
            x_t[0] += 2 * np.cos(math.radians(good_deg))
            x_t[1] += 2 * np.sin(math.radians(good_deg))
        
        if time > dots_on:
            x_t[-1] += self.D_mstim_strength

            y_t[:] += R
            
            if time >= dots_on + 100:
                mask_t[:] += (min(time, go) - dots_on)/(go - dots_on)
        
        if time > go:
            mask_t[:] = 3
        
        return x_t, y_t, mask_t
    
class MT_broadInSharpOut_withR(Task):

    def __init__(self, dt=10, tau=100, T=1200, N_batch=100, N_rec=100, 
                 in_noise=0.2, coh=[0.6], k_in=0.3, k_out=0.8, M_mstim_strength=0, 
                 fix_shown=None, fix_reward=None, catchP=0.0, fix_onset=None, test1ofEach=False):
        N_in = 72 + 3
        N_out = 72 + 2
        if test1ofEach:
            assert fix_reward is not None and fix_shown is not None
            assert N_batch == fix_shown.shape[0]*fix_reward.shape[0]
        super().__init__(N_in, N_out, dt, tau, T, N_batch)
        self.N_dirIn = N_in - 3
        self.N_dirOut = N_out - 2
        self.N_rec = N_rec
        self.in_noise = in_noise
        self.coh = coh
        self.k_in = k_in
        self.k_out = k_out
        self.M_mstim_strength = M_mstim_strength
        self.fix_shown = fix_shown
        self.fix_reward = fix_reward
        self.catchP = catchP
        self.fix_onset = fix_onset
        self.test1ofEach = test1ofEach
        
    def generate_trial_params(self, batch, trial):
        
        """"Define parameters for each trial.
    
        Using a combination of randomness, presets, and task attributes, define the necessary trial parameters.
    
        Args:
            batch (int): The batch number that this trial is part of.
            trial (int): The trial number of the trial within the batch.
    
        Returns:
            dict: Dictionary of trial parameters.
        """
        # ----------------------------------
        # Define parameters of a trial
        # ----------------------------------
            
        params = dict()
        N_shownDirs = self.N_dirOut
        
        if self.test1ofEach:
            good_deg = self.fix_reward[trial//self.fix_shown.shape[0]]
            shown_deg = self.fix_shown[trial%self.fix_shown.shape[0]]
        
        else:
            if self.fix_reward is not None:
                good_deg = rng.choice(self.fix_reward)
            else:
                good_deg = 20*rng.integers(0,18)
                
            if self.fix_shown is not None:
                shown_deg = rng.choice(self.fix_shown)
            else:
                if rng.uniform(0,1) < self.catchP:
                    shown_deg = None
                else:
                    shown_deg = (360/N_shownDirs) * rng.integers(0, N_shownDirs)
        
        params['good_deg'] = good_deg
        params['shown_deg'] = shown_deg
        
        if self.fix_onset is not None:
            params['t_ring'] = self.fix_onset[0]
            params['t_dots'] = self.fix_onset[1]
        else:
            params['t_ring'] = 10 * rng.integers(5, 15)
            params['t_dots'] = 10 * rng.integers(45, 55)
            
        if shown_deg is not None:
            params['coherence'] = rng.choice(self.coh)
        
        return params

    def trial_function(self, time, params):
        """ Compute the trial properties at the given time.
    
        Based on the params compute the trial stimulus (x_t), correct output (y_t), and mask (mask_t) at the given time.
    
        Args:
            time (int): The time within the trial (0 <= time < T).
            params (dict): The trial params produced generate_trial_params()
    
        Returns:
            tuple:
    
            x_t (ndarray(dtype=float, shape=(N_in,))): Trial input at time given params.
            y_t (ndarray(dtype=float, shape=(N_out,))): Correct trial output at time given params.
            mask_t (ndarray(dtype=bool, shape=(N_out,))): True if the network should train to match the y_t, False if the network should ignore y_t when training.
    
        """
        
        # ----------------------------------
        # Initialize with noise
        # ----------------------------------
        in_noise = self.in_noise
        x_t = np.zeros(self.N_in)
        x_t[:] += np.sqrt(2 * self.alpha * in_noise**2) * rng.standard_normal(self.N_in)
        
        y_t = np.zeros(self.N_out)
        mask_t = np.zeros(self.N_out)
    
        # ----------------------------------
        # Retrieve parameters
        # ----------------------------------
        good_deg = params['good_deg']
        t_ring = params['t_ring']
        t_dots = params['t_dots']
        
        shown_deg = params['shown_deg']
        if shown_deg is not None:
            coh = params['coherence']
        
        # ----------------------------------
        # Compute values
        # ----------------------------------

        mask_t[:] = 0.2
        
        if time >= t_ring:
            
            x_t[0] += 2 * np.cos(np.radians(good_deg))
            x_t[1] += 2 * np.sin(np.radians(good_deg))
            
            y_t[0] += np.cos(np.radians(good_deg))
            y_t[1] += np.sin(np.radians(good_deg))
            
            mask_t[:2] += 0.6 * (min(time, t_dots) - t_ring)/(t_dots - t_ring)
            
        if time >= t_dots:
        
            x_t[-1] += self.M_mstim_strength
            
            if shown_deg is not None:
                unit_dirs_in = np.arange(0, 360, 360//self.N_dirIn)
                unit_dirs_out = np.arange(0, 360, 360//self.N_dirOut)
            
                k_in = self.k_in
                k_out = self.k_out
                assert k_out > k_in
                A = 2.0
                assert A >= 1.0
                
                x_t[2:-1] += -0.3 + np.exp(
                    coh * k_in * np.cos(np.radians(unit_dirs_in - shown_deg)))
                
                y_t[2:] += A * (-0.3 + np.exp(
                    coh * k_out * np.cos(np.radians(unit_dirs_out - shown_deg))))
                
                if time >= t_dots + 100:
                    mask_t[2:] += 0.8 * (time - t_dots)/(self.T - t_dots)
                
                if time >= self.T-100:
                    mask_t[:] *= 2
                
        return x_t, y_t, mask_t
