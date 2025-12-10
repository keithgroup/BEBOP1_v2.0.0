# MIT License
# Copyright (c) 2025, Barbaro Zulueta, John A. Keith

"""Compute the distance matrix and the trig. projection functions

MakeSureZero :: makes the cosine equal to zero if it is less than 0.00001
CalculateSin :: calculate the sine of the function
Spatial_Properties :: Compute the distance matrix and angles
"""

import numpy as np

def MakeSureZero(CosTheta):
    """If the absolute CosTheta < 0.0001, then make it zero."""
    if np.abs(CosTheta) < 0.00001:
        CosTheta = np.float64(0)
    return CosTheta

def CalculateSin(CosTheta):
    """Calculate the sine of the theta using the value of cos(theta)"""
    SinTheta = np.sqrt(1 - CosTheta**2)
    if CosTheta < np.float64(0):
        SinTheta *= -1
    elif CosTheta >= np.float64(0):
        SinTheta *= 1
    return -SinTheta

def Spatial_Properties(XYZ):
    """Compute the distance matrix and the trig. projection functions
    
    Parameters
    ----------
    XYZ: np.ndarray
        Standard orientation cartesian coordinates.
    
    Returns
    ------
    tuple
        (DistanceMatrix, Trig): Distance matrix and trigonometric functions
    """
    
    length = XYZ.shape[0] 
    DistanceMatrix = np.zeros((length, length), dtype=np.float64)

    Cos = {'X': np.zeros((length, length), dtype=np.float64),
           'Y': np.zeros((length, length), dtype=np.float64),
           'Z': np.zeros((length, length), dtype=np.float64)}

    Sin = {'X': np.zeros((length, length), dtype=np.float64),
           'Y': np.zeros((length, length), dtype=np.float64),
           'Z': np.zeros((length, length), dtype=np.float64)}

    # Calculate all pairwise distances
    for i in range(length):
        for j in range(length):
            if i != j:
                # Calculate distance
                dx = XYZ[i][0] - XYZ[j][0]
                dy = XYZ[i][1] - XYZ[j][1] 
                dz = XYZ[i][2] - XYZ[j][2]
                
                d = np.sqrt(dx*dx + dy*dy + dz*dz)
                DistanceMatrix[i][j] = d
                
                # Calculate direction cosines
                if d > 0:
                    cos_x = MakeSureZero(dx / d)
                    cos_y = MakeSureZero(dy / d)
                    cos_z = MakeSureZero(dz / d)
                    
                    Cos['X'][i][j] = cos_x
                    Cos['Y'][i][j] = cos_y
                    Cos['Z'][i][j] = cos_z
                    
                    Sin['X'][i][j] = CalculateSin(cos_x)
                    Sin['Y'][i][j] = CalculateSin(cos_y) 
                    Sin['Z'][i][j] = CalculateSin(cos_z)
        
    Trig = (Cos['X'], Sin['X'], Cos['Y'], Sin['Y'], Cos['Z'], Sin['Z'])
    return (DistanceMatrix, Trig)
