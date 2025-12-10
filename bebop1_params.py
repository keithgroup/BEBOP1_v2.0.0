# MIT License
# 
# Copyright (c) 2025, Barbaro Zulueta, John A. Keith
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Parameters for the BEBOP code (version 2.0.0) - OPTIMIZED

BEBOP_Pair :: BEBOP atom-pair parameters (i.e., beta, zeta, R_sigma and D_AB) - WITH CACHING
BEBOP_Atom :: single atom parameters (i.e., n_2s, CBS-QB3 excitation energies from 1s to 2s) - WITH CACHING
"""

import numpy as np
from functools import lru_cache
from typing import Tuple, Union

# Pre-compute lookup tables for faster access
_ATOM_TO_INDEX = {'H': 0, 'He': 1, 'Li': 2, 'Be': 3, 'B': 4, 'C': 5, 'N': 6, 'O': 7, 'F': 8}

# Pre-computed parameter arrays (faster than dictionary lookups)
_BETA_MATRIX = np.array([
    [144.77,  0.000,  119.91, 143.19, 168.44, 178.45, 192.39, 258.42, 372.61],
    [0.000,  0.000,   0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000],
    [119.91,  0.000,   43.41,  86.36, 121.82, 178.60, 207.00, 342.56, 760.99],
    [143.19,  0.000,   86.36, 122.27, 147.88, 193.93, 217.45, 298.97, 474.24],
    [168.44,  0.000,  121.82, 147.88, 160.70, 201.30, 233.62, 316.25, 468.04],
    [178.45,  0.000,  178.60, 193.93, 201.30, 225.87, 233.16, 291.28, 403.33],
    [192.39,  0.000,  207.00, 217.45, 233.62, 233.16, 215.55, 271.38, 322.98],
    [258.42,  0.000,  342.56, 298.97, 316.25, 291.28, 271.38, 257.25, 252.10],
    [372.61,  0.000,  760.99, 474.24, 468.04, 403.33, 322.98, 252.10, 289.78]
])

_ZETA_MATRIX = np.array([
    [-8.19, -9.55, -2.71, -5.31, -5.92, -7.17, -7.81, -8.68,  -9.56],
    [-9.55, -9.55, -9.55, -9.55, -9.55, -9.55, -9.55, -9.55,  -9.55],
    [-2.71, -9.55, -3.26, -2.85, -5.77, -7.02, -4.85, -2.18,  -1.24],
    [-5.31, -9.55, -2.85, -4.28, -4.27, -4.26, -4.25, -4.25,  -4.99],
    [-5.92, -9.55, -5.77, -4.27, -6.73, -6.68, -6.64, -5.57,  -4.61],
    [-7.17, -9.55, -7.02, -4.26, -6.68, -7.53, -7.44, -7.30,  -7.57],
    [-7.81, -9.55, -4.85, -4.25, -6.64, -7.44, -13.84, -8.99, -7.50],
    [-8.68, -9.55, -2.18, -4.25, -5.57, -7.30, -8.99, -8.42, -10.91],
    [-9.56, -9.55, -1.24, -4.99, -4.61, -7.57, -7.50, -10.91, -1.17]
])

_R_EQ_MATRIX = np.array([
    [0.654,  1.000, 1.593, 1.327, 1.190, 1.091, 1.016, 0.962, 0.920],
    [1.000,  1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
    [1.593,  1.000, 2.705, 2.399, 2.183, 1.677, 1.569, 1.570, 1.560],
    [1.327,  1.000, 2.399, 2.080, 1.867, 1.673, 1.495, 1.518, 1.373],
    [1.190,  1.000, 2.183, 1.867, 1.727, 1.553, 1.388, 1.271, 1.324],
    [1.091,  1.000, 1.677, 1.673, 1.553, 1.532, 1.392, 1.200, 1.389],
    [1.016,  1.000, 1.569, 1.495, 1.388, 1.392, 1.095, 1.148, 1.430],
    [0.962,  1.000, 1.570, 1.518, 1.271, 1.200, 1.148, 1.207, 1.434],
    [0.920,  1.000, 1.560, 1.373, 1.324, 1.389, 1.430, 1.434, 1.408]
])

_D_MATRIX = np.array([
    [104.45, 0.000,  55.68,  92.28, 104.54, 103.60, 105.93, 117.73, 136.01],
    [0.000, 0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000],
    [55.68, 0.000,  24.03,  42.30,  44.48,  46.45,  72.02, 102.88, 136.27],
    [92.28, 0.000,  42.30,  71.67,  82.44,  92.03, 120.86, 147.18, 176.64],
    [104.54, 0.000,  44.48,  82.44, 136.32, 143.94, 177.14, 219.44, 169.55],
    [103.60, 0.000,  46.45,  92.03, 143.94, 226.85, 157.24, 179.47, 110.06],
    [105.93, 0.000,  72.02, 120.86, 177.14, 157.24, 122.65, 150.43,  69.31],
    [117.73, 0.000, 102.88, 147.18, 219.44, 179.47, 150.43, 119.71,  48.44],
    [136.01, 0.000, 136.27, 176.64, 169.55, 110.06,  69.31,  48.44,  37.44]
])

# Hybridization parameters as arrays for faster access
_HYBRID_N2S = np.array([0.000, 0.000, 1.000, 2.000, 2.000, 2.000, 2.000, 2.000, 2.000])
_HYBRID_E2S2P = np.array([0.0000, 0.000, 42.479, 65.008, 85.177, 98.098, 134.827, 172.055, 209.517])

@lru_cache(maxsize=128)
def BEBOP_Pair(Atom1: str, Atom2: str, res_strain: bool = False) -> Union[float, Tuple[float, float, float, float]]:
    """Fitted atom-pair parameters used in the BEBOP equation - OPTIMIZED WITH CACHING

    Parameters
    ----------
    Atom1 : str
        Name of the atom (i.e, 'H','He',etc.)
    Atom2 : str
        Name of the atom interacting with Atom1 (i.e,'H','He',etc.)
    res_strain : bool, optional
        Need the beta parameter only for resonance and strain calculations
    
    Returns 
    -------
    beta : float
        Fixed parameter used compute the extended-Hückel bond energy (with ZPE)  
    zeta : float
        Slater-type exponential parameter for the short-range repulsion
    R_sigma : float
        Classical turning distance (i.e., :math:`E_tot \approx E_short`)  
    D_AB : float
        Bond dissociation energy parameter (with ZPE) 
    """
    
    # Fast lookup using pre-computed indices
    try:
        idx1 = _ATOM_TO_INDEX[Atom1]
        idx2 = _ATOM_TO_INDEX[Atom2]
    except KeyError as e:
        raise ValueError(f"Unknown atom: {e}")
    
    betaAB = _BETA_MATRIX[idx1, idx2]
    
    if res_strain:
        return betaAB
    else:
        zetaAB = _ZETA_MATRIX[idx1, idx2]
        R_sigma = _R_EQ_MATRIX[idx1, idx2] / np.sqrt(2)
        D_AB = _D_MATRIX[idx1, idx2]
        return (betaAB, zetaAB, R_sigma, D_AB)

@lru_cache(maxsize=32)
def BEBOP_Atom(Atom: str) -> Tuple[float, float]:
    """Single-atom parameters used in the BEBOP equation - OPTIMIZED WITH CACHING
    
    Parameters
    ----------
    Atom : str
        The name of the atom (i.e, 'H','He',etc.)  
    
    Returns
    -------
    tuple
        (number of 2s electrons, UCBS-QB3 excitation energy from 2s to 2p) in Atom object.
    """
    
    try:
        idx = _ATOM_TO_INDEX[Atom]
        return (_HYBRID_N2S[idx], _HYBRID_E2S2P[idx])
    except KeyError:
        raise ValueError(f"Unknown atom: {Atom}")

# Vectorized versions for batch operations
def BEBOP_Pair_vectorized(atoms1: np.ndarray, atoms2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized version of BEBOP_Pair for multiple atom pairs"""
    
    indices1 = np.array([_ATOM_TO_INDEX[atom] for atom in atoms1])
    indices2 = np.array([_ATOM_TO_INDEX[atom] for atom in atoms2])
    
    beta = _BETA_MATRIX[indices1, indices2]
    zeta = _ZETA_MATRIX[indices1, indices2]
    r_sigma = _R_EQ_MATRIX[indices1, indices2] / np.sqrt(2)
    d_ab = _D_MATRIX[indices1, indices2]
    
    return beta, zeta, r_sigma, d_ab

def BEBOP_Atom_vectorized(atoms: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized version of BEBOP_Atom for multiple atoms"""
    
    indices = np.array([_ATOM_TO_INDEX[atom] for atom in atoms])
    
    n2s = _HYBRID_N2S[indices]
    e2s2p = _HYBRID_E2S2P[indices]
    
    return n2s, e2s2p
