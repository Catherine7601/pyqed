#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 10:28:26 2025

Ehrenfest dynamcis with Shin-Metiu moddel 

@author: Bing Gu (gubing@westlake.edu.cn)
"""

from pyqed.models.ShinMetiu import ShinMetiu2
import numpy as np

from pyqed.namd.ehrenfest import Ehrenfest

mol = ShinMetiu2()

mol.build(domain=[[-10, 10], ] * 2, npts=[31, 31])

ed = Ehrenfest(ndim=mol.ndim, ntraj=10, nstates=mol.nstates, model=mol, mass=[1836, ] * 2)

ed.run(dt=0.1, nt=10)




# # slice the electronic data to see if it makes sense
# x = np.linspace(-2,2, 50)
# nx = len(x)
# e = np.zeros((nx, mol.nstates))
# grad = np.zeros((nx, mol.nstates, mol.ndim))
# nac = np.zeros((nx, mol.nstates, mol.nstates, mol.ndim))

# for i in range(len(x)):
#     R = [x[i], 0]
    
#     e[i], grad[i], nac[i] = mol.nonadiabatic_coupling(R)


# import ultraplot as plt
# fig, ax = plt.subplots()

# for i in range(mol.nstates):
#     ax.plot(x, e[:,i])

# fig, ax = plt.subplots()
# for i in range(mol.nstates):
#     ax.plot(x, grad[:,i, 0])
#     ax.plot(x, grad[:,i, 1])

