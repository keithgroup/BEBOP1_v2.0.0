import numpy as np
import bebop1_params as par

def BEBOP1(DistanceMatrix, AtomSym, MolOcc2s, MullikenPop):
    """Compute the total BEBOP1 energy (SCF+ZPVE) at 0 K.  To remove ZPVE contributions subtract ZPVE from a B3LYP/CBSB7 calculation"""
    
    BEBOP = 0
    SIZE = AtomSym.shape[0]
    
    # Main loop for pair interactions
    for l in range(1, SIZE):
        for n in range(l):
            betaAB, zetaAB, R_sigma, D_AB = par.BEBOP_Pair(AtomSym[l], AtomSym[n])
            
            # Covalent energy term
            covalent_term = -2 * betaAB * MullikenPop[l][n]
            BEBOP += covalent_term
            
            # Repulsion energy term
            distance = DistanceMatrix[l][n]
            if distance > 0:
                repulsion_term = D_AB * np.exp(zetaAB * (distance - R_sigma))
                BEBOP += repulsion_term
    
    # Calculate the hybridization energy
    hybridization_energy = 0
    n = 0
    for i, atom in enumerate(AtomSym):
        if atom in ['H', 'He']:
            continue
        else:
            n2s, E2s2p = par.BEBOP_Atom(atom)
            
            if isinstance(MolOcc2s, np.ndarray):
                if n < len(MolOcc2s):
                    occ2s_val = MolOcc2s[n]
                else:
                    occ2s_val = 0.0
                n += 1
            else:
                occ2s_val = MolOcc2s
            
            hybrid_term = (n2s - occ2s_val) * E2s2p
            hybridization_energy += hybrid_term
    
    BEBOP += hybridization_energy
    return BEBOP

def BEBOPBondEnergies_BEBOP1(MullikenPop, DistanceMatrix, AtomSym, MolOcc2s, EROCBSQB3=None):
    """Calculate bond energies for BEBOP1 - CLEAN VERSION"""
    
    BEBOP = 0
    SIZE = AtomSym.shape[0]
    Ecov = np.zeros((SIZE, SIZE))
    
    # Main bond energy calculation
    for l in range(1, SIZE):
        for n in range(l):
            betaAB, zetaAB, R_sigma, D_AB = par.BEBOP_Pair(AtomSym[l], AtomSym[n])
            
            # Total covalent energy (no sigma/pi split)
            Ecov[l][n] = -2 * betaAB * MullikenPop[l][n]
            BEBOP += -2 * betaAB * MullikenPop[l][n]
            
            # Repulsion energy
            distance = DistanceMatrix[l][n]
            if distance > 0:
                rep_energy = D_AB * np.exp(zetaAB * (distance - R_sigma))
                BEBOP += rep_energy
                Ecov[l][n] += rep_energy
    
    # Hybridization energy
    Ehybrid = np.zeros(SIZE)
    hybridization_energy = 0
    n = 0
    for i, atom in enumerate(AtomSym):
        if atom in ['H', 'He']:
            continue
        else:
            n2s, E2s2p = par.BEBOP_Atom(atom)
            
            if isinstance(MolOcc2s, np.ndarray):
                if n < len(MolOcc2s):
                    occ2s_val = MolOcc2s[n]
                else:
                    occ2s_val = 0.0
                n += 1
            else:
                occ2s_val = MolOcc2s
            
            Ehybrid[i] = (n2s - occ2s_val) * E2s2p
            hybridization_energy += Ehybrid[i]
            
    BEBOP += hybridization_energy
    
    # For BEBOP1, sigma and pi are the same (total covalent)
    Esig = Ecov.copy()  # All covalent energy counted as "sigma" 
    Epi = np.zeros((SIZE, SIZE))  # No separate pi component in BEBOP1
    
    # Net energies with hybridization corrections
    TBE = np.zeros(SIZE)
    for i in range(1, SIZE):
        for j in range(i):
            TBE[i] += Ecov[i][j]
            TBE[j] += Ecov[i][j]
    
    Enet = np.zeros((SIZE, SIZE))
    Enet_sig = np.zeros((SIZE, SIZE)) 
    Enet_pi = np.zeros((SIZE, SIZE))
    
    for i in range(1, SIZE):
        for j in range(i):
            if TBE[i] != 0 and TBE[j] != 0:
                Factor = 1.0 + Ehybrid[i] / TBE[i] + Ehybrid[j] / TBE[j]
                Enet[i][j] = Ecov[i][j] * Factor
                Enet_sig[i][j] = Ecov[i][j] * Factor  # All net energy is "sigma"
                Enet_pi[i][j] = 0.0  # No pi component
    
    # Composite table
    CompositeTable = np.zeros((SIZE, SIZE))
    for i in range(SIZE):
        for j in range(SIZE):
            if i < j:
                CompositeTable[j][i] = Ecov[i][j]
            elif i > j:
                CompositeTable[i][j] = Enet[i][j]
    np.fill_diagonal(CompositeTable, Ehybrid)
    
    if EROCBSQB3 is None:
        return (Esig, Epi, Ecov, Enet_sig, Enet_pi, Enet, CompositeTable, BEBOP)
    else:
        if BEBOP != 0:
            Enet_RN = Enet * EROCBSQB3 / BEBOP
        else:
            Enet_RN = Enet
        return (Esig, Epi, Ecov, Enet_sig, Enet_pi, Enet, CompositeTable, Enet_RN, BEBOP)

def BEBOPBondEnergies(SigmaBondOdr, PiBondOdr, MullikenPop, DistanceMatrix, AtomSym, MolOcc2s, EROCBSQB3=None):
    """Dispatcher for BEBOP1 bond energies - CLEAN VERSION"""
    return BEBOPBondEnergies_BEBOP1(MullikenPop, DistanceMatrix, AtomSym, MolOcc2s, EROCBSQB3)

def resonance(Data):
    """Compute the resonance energy of the aromatic system (in kcal/mol)"""
    res_E = 0
    for l in Data.keys():
        BO_tot, n, ref_BO, atoms = Data[l] 
        betaAB = par.BEBOP_Pair(atoms[0], atoms[1], res_strain=True) 
        res_E += (BO_tot - int(n) * ref_BO) * betaAB  
    return res_E
    
def strain(Data):
    """Compute the ring strain energy for a ring (in kcal/mol)."""
    strain_E = 0
    betaAB = par.BEBOP_Pair('C', 'C', res_strain=True)
    for l in Data.keys():
        BO_tot, n, ref_BO = Data[l] 
        strain_E += (int(n) * ref_BO - BO_tot) * betaAB 
    return strain_E
