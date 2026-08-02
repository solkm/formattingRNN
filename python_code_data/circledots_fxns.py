#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 14:48:13 2022

@author: Sol
"""

import numpy as np
import math

def distance_deg_arr(arr, deg):
    assert np.all(arr>=0) and np.all(arr<360)
    assert deg>=0 and deg<360
    diff = np.abs(arr-deg)
    diff[diff>180] = 360 - diff[diff>180]
    return diff

def distance_deg(deg1, deg2):
    diff = np.abs((deg1 - deg2)%360)
    if diff > 180:
        diff = 360 - diff
    return diff
    
def signed_distance_deg(deg1, deg2):
    diff = deg1 - deg2
    diff = diff - (diff + 180)//360 * 360
    return diff

def distance_to_degloc(deg2, diff):
    assert (deg2 < 360) and (deg2 >= 0), 'angle must be in the interval [0, 360) deg'
    assert np.abs(diff) <= 180, 'enter a distance between -180 and 180'
    
    deg1 = deg2 + diff
    
    if deg1>=360:
        deg1 = deg1%360
    elif deg1<0:
        deg1 = 360 + deg1
    return deg1

def between_or_outside(start, end, angle):
    assert angle>=0 and angle<360, 'invalid angle'
    
    start2end = signed_distance_deg(end, start)
    start2angle = signed_distance_deg(angle, start)
    lowerb = min(start2end, 0)
    upperb = max(start2end, 0)
    
    return (start2angle >= lowerb and start2angle <= upperb)

def avg_angles_deg(angles):
    angles_rad = np.radians(angles)
    avg_deg =  np.degrees( np.arctan2( np.sum(np.sin(angles_rad)) , np.sum(np.cos(angles_rad)) ) )
    if avg_deg<0:
        avg_deg += 360
    return avg_deg
    
def rel2center(degs, center):
    reldegs = degs - center
    reldegs[reldegs>180] -= 360
    reldegs[reldegs<=-180] += 360
    return reldegs