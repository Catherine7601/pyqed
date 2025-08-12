#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 19:56:20 2025

@author: Bing Gu

@email: gubing@westlake.edu.cn
"""

# import psi4
import numpy as np
from pyqed import au2ev
from pyqed.qchem.ci import CI
from opt_einsum import contract
from scipy.sparse.linalg import eigsh

class CIS(CI):
    def run(self, nroots=1):
        return kernel(self.mf, nroots)

    def make_rdm1(self):
        pass

# set molecule
# mol = psi4.geometry("""
# o
# h 1 1.0
# h 1 1.0 2 104.5
# symmetry c1
# """)

# # set options
# psi4.set_options({'basis': 'sto-3g',
#                   'scf_type': 'pk',
#                   'e_convergence': 1e-8,
#                   'd_convergence': 1e-8})

# # compute the Hartree-Fock energy and wave function
# scf_e, wfn = psi4.energy('SCF', return_wfn=True)


# Grab data from wavfunction

def kernel(mf, nroots=1):
    # number of doubly occupied orbitals
    # nocc   = wfn.nalpha()
    nocc = mf.nocc

    # total number of orbitals
    # nmo     = wfn.nmo()
    nmo = mf.nmo

    # number of virtual orbitals
    nvir   = nmo - nocc

    # orbital energies
    eps     = mf.mo_energy # np.asarray(wfn.epsilon_a())

    # occupied orbitals:
    # Co = wfn.Ca_subset("AO", "OCC")
    Co = mf.mo_coeff[:, :nocc]

    # virtual orbitals:
    # Cv = wfn.Ca_subset("AO", "VIR")
    Cv = mf.mo_coeff[:, nocc:]

    # use Psi4's MintsHelper to generate ERIs
    # mints = psi4.core.MintsHelper(wfn.basisset())

    eri = mf.mol.eri

    # build the (ov|ov) integrals:
    # ovov = np.asarray(eri(Co, Cv, Co, Cv))
    ovov = contract('pqrs, pi, qa, rj, sb -> iajb', eri, Co, Cv, Co, Cv)


    # build the (oo|vv) integrals:
    oovv = contract('pqrs, pi, qj, ra, sb -> ijab', eri, Co, Co, Cv, Cv)

    # strip out occupied orbital energies, eps_o spans 0..ndocc-1
    eps_o = eps[:nocc]

    # strip out virtual orbital energies, eps_v spans 0..nvirt-1
    eps_v = eps[nocc:]

    # CIS Hamiltonian
    H = np.zeros((nocc*nvir, nocc*nvir))

    # build singlet hamiltonian
    for i in range(0,nocc):
        for a in range(0,nvir):
            ia = i * nvir + a
            for j in range(0,nocc):
                for b in range(0,nvir):
                    jb = j * nvir + b
                    H[ia][jb] = 2.0 * ovov[i][a][j][b] - oovv[i][j][a][b]
            H[ia][ia] += eps_v[a] - eps_o[i]

    print(H)

    # diagonalize Hamiltonian
    eig, ci = eigsh(H, k=nroots, which='SA')


    print("")
    print("    ==> CIS singlet excitation energies (eV) <==")
    print("")
    for ia in range(0,nroots):
            print("    %5i %10.6f" % (ia,eig[ia]*au2ev))
    print("")

    # build triplet hamiltonian
    for i in range(0,nocc):
        for a in range(0,nvir):
            ia = i * nvir + a
            for j in range(0,nocc):
                for b in range(0, nvir):
                    jb = j * nvir + b
                    H[ia][jb] = - oovv[i][j][a][b]
            H[ia][ia] += eps_v[a] - eps_o[i]

    # diagonalize Hamiltonian
    eig, ci = eigsh(H, k=nroots, which='SA')

    print("")
    print("    ==> CIS triplet excitation energies (eV) <==")
    print("")
    for ia in range(0,nroots):
        print("    %5i %10.4f" % (ia,eig[ia]*au2ev))
    print("")
    return eig, ci


if __name__=='__main__':
    from pyqed.qchem import Molecule
    from pyscf import gto
    # mol = Molecule(atom = [
    #     ['H' , (0. , 0. , 0)],
    #     ['Li' , (0. , 0. , 2)], ])

    # set molecule
    atom = """
    o
    h 1 1.0
    h 1 1.0 2 104.5
    """
    mol = gto.M(atom)

    mol = Molecule(mol._atom)


    mol.basis = 'sto3g'
    mol.charge = 0
    # mol.unit = 'b'
    mol.build()

    mf = mol.RHF().run()
    print(mol.nelec)
    ci = CIS(mf)
    ci.run(nroots=6)