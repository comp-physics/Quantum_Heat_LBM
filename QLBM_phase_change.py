#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:22:36 2024

@author: cjawetz3
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import Permutation
from qiskit.result import marginal_counts
import math
from math import pi, sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.optimize import fsolve
from scipy import special
from scipy.stats import linregress
import pandas as pd
from scipy.interpolate import interp1d
from qiskit_algorithms.phase_estimators import IterativePhaseEstimation
from qiskit.primitives import BackendSampler 
from qiskit.circuit.library import UnitaryGate

plt.rc('axes', labelsize=9.)    
plt.rc('xtick', labelsize=9.)    
plt.rc('ytick', labelsize=9.)   

backend = 'aer_simulator_matrix_product_state' 
sampler = BackendSampler(backend)
ipe = IterativePhaseEstimation(num_iterations=5, sampler=sampler)

# D1Q3 lattice constants
w = np.array([1/6, 2/3, 1/6]) # weight coeffecients
cx = np.array([1, 0, -1])   #lattice velocities
csq = 1/3   #square of lattice speed
ux = 0.  # advection x-velocity
tm=0.4
cp=1
la=10
tb=1
to=0
dx=1
dt=1

#omega=1
alpha=(1/6)*(2-dt)

def f(lambd):       #calculates gamma for analytical solution
    val1=cp*(tb-tm)/la/(np.exp(lambd**2)*special.erf(lambd))
    val2=(cp*(tm-to)/la)/(np.exp(lambd**2)*special.erfc(lambd))
    val3=lambd*np.sqrt(np.pi)
    return val1-val2-val3


def compute_feq(rho, w, cx, ux, csq):       #equilibrium distribution
    feq = np.zeros((3,M))
    for i in range(3):
        feq[i] = w[i] * (1 + cx[i]*ux/csq) * rho
    return feq

def ini(x, w, cx, ux, csq):                 #initialize problem with T_bound on the left side
    M = len(x)
    rho = np.zeros(M)   
    rho[0] = 1
    feq = compute_feq(rho, w, cx, ux, csq)
    f = feq
    state=np.zeros(M)
    state[0]=1
    return f, rho, state

def update_encoding(qc, f, M):
    for k in range(M): 
        amp000 = np.sqrt( (1-f[0][k]) * (1-f[1][k])* (1-f[2][k]) )
        amp100 = np.sqrt( (1-f[1][k]) * (1-f[2][k]) * f[0][k] )
        amp010 = np.sqrt( (1-f[0][k]) * (1-f[2][k]) * f[1][k] )        
        amp001 = np.sqrt( (1-f[0][k]) * (1-f[1][k]) * f[2][k] )
        amp011 = np.sqrt( (1-f[0][k]) * f[1][k] * f[2][k] )
        amp101 = np.sqrt( (1-f[1][k]) * f[0][k] * f[2][k] )
        amp110 = np.sqrt( (1-f[2][k]) * f[0][k] * f[1][k] )
        amp111 = np.sqrt( f[0][k] * f[1][k] * f[2][k] )
        vector = np.array([amp111, amp110, amp101, amp011, amp001, amp010, amp100, amp000])
        qc.prepare_state(vector, [0+3*k, 1+3*k, 2+3*k])  
    return qc

def update_encoding_rate(qc, f, M):         #leave ancilla qubits out of this
    for k in range(M-rate): 
        amp000 = np.sqrt( (1-f[0][k]) * (1-f[1][k])* (1-f[2][k]) )
        amp100 = np.sqrt( (1-f[1][k]) * (1-f[2][k]) * f[0][k] )
        amp010 = np.sqrt( (1-f[0][k]) * (1-f[2][k]) * f[1][k] )        
        amp001 = np.sqrt( (1-f[0][k]) * (1-f[1][k]) * f[2][k] )
        amp011 = np.sqrt( (1-f[0][k]) * f[1][k] * f[2][k] )
        amp101 = np.sqrt( (1-f[1][k]) * f[0][k] * f[2][k] )
        amp110 = np.sqrt( (1-f[2][k]) * f[0][k] * f[1][k] )
        amp111 = np.sqrt( f[0][k] * f[1][k] * f[2][k] )
        vector = np.array([amp111, amp110, amp101, amp011, amp001, amp010, amp100, amp000])
        qc.prepare_state(vector, [0+3*k, 1+3*k, 2+3*k])  
    return qc

#collision matrix
U = [[0, 0, 0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 1j/sqrt(3), np.exp(-1j*pi/6)/sqrt(3), np.exp(-1j*pi/6)/sqrt(3), 0],
     [0, 0, 0, 0, np.exp(-1j*pi/6)/sqrt(3), 1j/sqrt(3), np.exp(-1j*pi/6)/sqrt(3), 0],
     [0, np.exp(-1j*pi/6)/sqrt(3), np.exp(-1j*pi/6)/sqrt(3), 1j/sqrt(3), 0, 0, 0, 0],
     [0, 0, 0, 0, np.exp(-1j*pi/6)/sqrt(3), np.exp(-1j*pi/6)/sqrt(3), 1j/sqrt(3), 0],
     [0, np.exp(-1j*pi/6)/sqrt(3), 1j/sqrt(3), np.exp(-1j*pi/6)/sqrt(3), 0, 0, 0, 0],
     [0, 1j/sqrt(3), np.exp(-1j*pi/6)/sqrt(3), np.exp(-1j*pi/6)/sqrt(3), 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0]]

U_mat = np.array(U, dtype=complex)

U_gate = UnitaryGate(U_mat, label="U")


def collision_Diffusion(circ, k):
    circ.unitary(U, [0+3*k,1+3*k,2+3*k])
    return circ

def classical_streaming_map(n):
    
    if n < 6 or n % 3 != 0:
        raise ValueError("n must be at least 6 and a multiple of 3")

    mapping = {i: i for i in range(n)}
    
    # Apply the streaming pattern
    pattern = computeStreamingPattern(n)
    for i in range(0, len(pattern), 3):
        a, b, c = pattern[i:i+3]
        
    
    return mapping

def get_sum_qubit_indices(n, original_indices):
    mapping = classical_streaming_map(n)
    return [mapping[i] for i in original_indices]


def computeStreamingPattern(n):
    if (n >= 6):  #minimum for streaming is 2 sites, corresonding to 6qubits
        #first pair of qubits
        streamingPattern = [2, 1, 5]
        for i in range(3,n-4):
            if i%3 == 0:  
                streamingPattern.extend([i-3, i+1, i+5])    #left, stationary, right
        #last pair of qubits
        streamingPattern.extend([n-6, n-2, n-3])
    else:
        streamingPattern = []
    return streamingPattern

def classical_streaming(M, state, qubit_counts, numberOfShots):
    #read post collision probabilities
    f = np.zeros((3,M))
    fout = np.zeros((3,M))
    for k in range(M):
        if '1' in qubit_counts[2+3*k]:
            fout[0][k] = qubit_counts[2+3*k]['1'] / numberOfShots
        if '1' in qubit_counts[1+3*k]:
            fout[1][k] = qubit_counts[1+3*k]['1'] / numberOfShots
        if '1' in qubit_counts[0+3*k]:
            fout[2][k] = qubit_counts[0+3*k]['1'] / numberOfShots
    
    # classical streaming
    for k in range(1,M):    #right 
        f[0][k] = fout[0][k-1]

    for k in range(M):     #stationary 
        f[1][k] = fout[1][k] 
        
    for k in range(M-1):     #left
        f[2][k] = fout[2][k+1]

    for i in range(M):
            if np.sum(f,axis=0)[i]>tm:
                if state[i]<1:
                    state[i]+=(np.sum(f,axis=0)[i]-tm)*cp/la
                    column_sum=(np.sum(f,axis=0)[i])                    
                    f[0][i]*=tm/column_sum
                    f[1][i]*=tm/column_sum
                    f[2][i]*=tm/column_sum
                if state[i]>=1:
                    column_sum=(np.sum(f,axis=0)[i])
                    f[0][i]+=f[0][i]*(state[i]-1)*la/cp/column_sum
                    f[1][i]+=f[1][i]*(state[i]-1)*la/cp/column_sum
                    f[2][i]+=f[2][i]*(state[i]-1)*la/cp/column_sum
                    state[i]=1

    f[0][0] = 1-f[2][0]-f[1][0]
    f[0][M-1]=0
    f[1][M-1]=0
    f[2][M-1]=0

    return f,state

def oneTimeStep_classicalStreaming(f, M, state, numberOfShots, backend):
    qc = QuantumCircuit(3*M)
    #step1: encoding
    qc = update_encoding(qc, f, M)
    qc.barrier()
    
    #step2: collision
    for k in range(M):
        qc = collision_Diffusion(qc, k)
        
    #step3: measurement
    qc.measure_all()
    job = transpile(qc, backend)
    result=backend.run(job, shots=numberOfShots).result()
    counts = result.get_counts(0)
    qubit_counts = [marginal_counts(counts, [qubit]) for qubit in range(3*M)]

    #step4: streaming
    f,state = classical_streaming(M, state, qubit_counts, numberOfShots)
    return f, qc,state

def oneTimeStep_quantumStreaming(f, M, numberOfShots, backend, t, maxT,bc,bcl,bcr):
    qc = QuantumCircuit(3*M+rate)
    #step1: encoding
    qc = update_encoding_rate(qc, f, M+rate)
    qc.barrier()
    
    #step2: collision
    for k in range(M):
        qc = collision_Diffusion(qc, k)
        
    
    
    #step3: streaming    
    qc.append(Permutation(num_qubits = 3*M, pattern = computeStreamingPattern(3*M)), range(3*M))
    
    boundary_indices = [0, 1, 2] 

    #qc.ry(angle,3*M+np.mod(t,rate))
    qc.mcry(2*np.arcsin(np.sqrt(1)),boundary_indices,3*M++np.mod(t,rate))
    qc.x(boundary_indices[0])
    qc.mcry(2*np.arcsin(np.sqrt(2/3)),boundary_indices,3*M++np.mod(t,rate))
    qc.x(boundary_indices[1])
    qc.mcry(2*np.arcsin(np.sqrt(1/3)),boundary_indices,3*M++np.mod(t,rate))
    qc.x(boundary_indices[0])
    qc.mcry(2*np.arcsin(np.sqrt(2/3)),boundary_indices,3*M++np.mod(t,rate))
    qc.x(boundary_indices[2])
    qc.mcry(2*np.arcsin(np.sqrt(1/3)),boundary_indices,3*M++np.mod(t,rate))
    qc.x(boundary_indices[0])
    qc.mcry(0,boundary_indices,3*M++np.mod(t,rate))
    qc.x(boundary_indices[1])
    qc.mcry(2*np.arcsin(np.sqrt(1/3)),boundary_indices,3*M++np.mod(t,rate))
    qc.x(boundary_indices[0])
    qc.mcry(2*np.arcsin(np.sqrt(2/3)),boundary_indices,3*M++np.mod(t,rate))
    qc.x(boundary_indices[2])

    qc.measure_all()
    job = transpile(qc, backend)
    result=backend.run(job, shots=numberOfShots).result()
    counts = result.get_counts(0)
    qubit_counts = [marginal_counts(counts, [qubit]) for qubit in range(3*M)]
    ex_counts=marginal_counts(counts, indices=[int(3*M+np.mod(t,rate))])
    try:
        ex_q=(ex_counts['1']/numberOfShots)*3-tm
    except KeyError as e:
        ex_q=-tm

    #read post streaming probabilities
    fout = np.zeros((3,M))
    for k in range(M):
        if '1' in qubit_counts[2+3*k]:
            fout[0][k] = qubit_counts[2+3*k]['1'] / numberOfShots
        if '1' in qubit_counts[1+3*k]:
            fout[1][k] = qubit_counts[1+3*k]['1'] / numberOfShots
        if '1' in qubit_counts[0+3*k]:
            fout[2][k] = qubit_counts[0+3*k]['1'] / numberOfShots
    f = fout

    if np.mod(t,min(rate,short))==0 or t<=2*rate:
        f[2][0] = bc-f[1][0]-f[0][0]
        f[0][M-1] = f[0][M-2]
        f[1][M-1] = 0
        f[2][M-1] = 0
    else:
        f[2][0]=bcl[2]    
        f[1][0]=bcl[1]    
        f[0][0]=bcl[0]    
        f[0][M-1] = f[0][M-2]
        f[1][M-1] = 0
        f[2][M-1] = 0
    return f, qc, ex_q


def multiTimeStep_quantumStreaming(f, M, numberOfShots, backend, t, maxT,bc,bcl,bcr):
    qc = QuantumCircuit(3*M+rate)
    #step1: encoding
    
    qc = update_encoding_rate(qc, f, M+rate)
    qc.barrier()
    for kk in range(rate):
        #step2: collision
        for k in range(M):
            qc = collision_Diffusion(qc, k)
        
        #step3: streaming    
        qc.append(Permutation(num_qubits = 3*M, pattern = computeStreamingPattern(3*M)), range(3*M))
        
        boundary_indices = [0, 1, 2] 
    
        #qc.ry(angle,3*M+np.mod(t,rate))
        qc.mcry(2*np.arcsin(np.sqrt(1)),boundary_indices,3*M+kk)
        qc.x(boundary_indices[0])
        qc.mcry(2*np.arcsin(np.sqrt(2/3)),boundary_indices,3*M+kk)
        qc.x(boundary_indices[1])
        qc.mcry(2*np.arcsin(np.sqrt(1/3)),boundary_indices,3*M+kk)
        qc.x(boundary_indices[0])
        qc.mcry(2*np.arcsin(np.sqrt(2/3)),boundary_indices,3*M+kk)
        qc.x(boundary_indices[2])
        qc.mcry(2*np.arcsin(np.sqrt(1/3)),boundary_indices,3*M+kk)
        qc.x(boundary_indices[0])
        qc.mcry(0,boundary_indices,3*M+kk)
        qc.x(boundary_indices[1])
        qc.mcry(2*np.arcsin(np.sqrt(1/3)),boundary_indices,3*M+kk)
        qc.x(boundary_indices[0])
        qc.mcry(2*np.arcsin(np.sqrt(2/3)),boundary_indices,3*M+kk)
        qc.x(boundary_indices[2])
    x
        qc.reset(0)
        qc.reset(1)
        qc.reset(2)
        qc.reset(M-3)
        qc.reset(M-2)
        qc.reset(M-1)
        
        qc.barrier()
    
        qc.ry(np.sqrt(bcl[0]),0)
        qc.ry(np.sqrt(bcl[1]),1)
        qc.ry(np.sqrt(bcl[2]),2)
        qc.ry(np.sqrt(bcr[0]),3*M-3)
        qc.ry(np.sqrt(bcr[1]),3*M-2)
        qc.ry(np.sqrt(bcr[2]),3*M-1)
    
    qc.measure_all()
    job = transpile(qc, backend)
    result=backend.run(job, shots=numberOfShots).result()
    counts = result.get_counts(0)
    qubit_counts = [marginal_counts(counts, [qubit]) for qubit in range(3*M)]
    ex_counts=marginal_counts(counts, indices=[int(3*M+np.mod(t,rate))])
    try:
        ex_q=(ex_counts['1']/numberOfShots)*3-tm
    except KeyError as e:
        ex_q=-tm

    #read post streaming probabilities
    fout = np.zeros((3,M))
    for k in range(M):
        if '1' in qubit_counts[2+3*k]:
            fout[0][k] = qubit_counts[2+3*k]['1'] / numberOfShots
        if '1' in qubit_counts[1+3*k]:
            fout[1][k] = qubit_counts[1+3*k]['1'] / numberOfShots
        if '1' in qubit_counts[0+3*k]:
            fout[2][k] = qubit_counts[0+3*k]['1'] / numberOfShots
    f = fout
    if np.mod(t,min(rate,short))==0 or t<=2*rate:
        f[2][0] = bc-f[1][0]-f[0][0]
        f[0][M-1] = f[0][M-2]
        f[1][M-1] = 0
        f[2][M-1] = 0
    else:
        f[2][0]=bcl[2]    
        f[1][0]=bcl[1]    
        f[0][0]=bcl[0]    
        f[0][M-1] = f[0][M-2]
        f[1][M-1] = 0
        f[2][M-1] = 0
    return f, qc, ex_q

def oneTimeStep_quantumStreaming_liq(f, M, numberOfShots, backend, t, maxT,bc,bcl,bcr):
    qc = QuantumCircuit(3*M+rate+1)
    #step1: encoding
    qc = update_encoding_rate(qc, f, M+rate)

    for kk in range(rate):

        #step2: collision
        for k in range(M):
            qc = collision_Diffusion(qc, k)
            
        
        #step3: streaming    
        qc.append(Permutation(num_qubits = 3*M, pattern = computeStreamingPattern(3*M)), range(3*M))
        
        boundary_indices = [3*M-3,3*M-2,3*M-1]
    
        #qc.ry(angle,3*M+np.mod(t,rate))
        qc.mcry(2*np.arcsin(np.sqrt(1)),boundary_indices,3*M++np.mod(t,rate))
        qc.x(boundary_indices[0])
        qc.mcry(2*np.arcsin(np.sqrt(2/3)),boundary_indices,3*M++np.mod(t,rate))
        qc.x(boundary_indices[1])
        qc.mcry(2*np.arcsin(np.sqrt(1/3)),boundary_indices,3*M++np.mod(t,rate))
        qc.x(boundary_indices[0])
        qc.mcry(2*np.arcsin(np.sqrt(2/3)),boundary_indices,3*M++np.mod(t,rate))
        qc.x(boundary_indices[2])
        qc.mcry(2*np.arcsin(np.sqrt(1/3)),boundary_indices,3*M++np.mod(t,rate))
        qc.x(boundary_indices[0])
        qc.mcry(0,boundary_indices,3*M++np.mod(t,rate))
        qc.x(boundary_indices[1])
        qc.mcry(2*np.arcsin(np.sqrt(1/3)),boundary_indices,3*M++np.mod(t,rate))
        qc.x(boundary_indices[0])
        qc.mcry(2*np.arcsin(np.sqrt(2/3)),boundary_indices,3*M++np.mod(t,rate))
        qc.x(boundary_indices[2])
    
        # qc.reset(0)
        # qc.reset(1)
        # qc.reset(2)
        # qc.reset(M-3)
        # qc.reset(M-2)
        # qc.reset(M-1)
        
        # qc.barrier()
    
        # qc.ry(np.sqrt(bcl[0]),0)
        # qc.ry(np.sqrt(bcl[1]),1)
        # qc.ry(np.sqrt(bcl[2]),2)
        # qc.ry(np.sqrt(bcr[0]),3*M-3)
        # qc.ry(np.sqrt(bcr[1]),3*M-2)
        # qc.ry(np.sqrt(bcr[2]),3*M-1)
        

    qc.measure_all()
    job = transpile(qc, backend)
    result=backend.run(job, shots=numberOfShots).result()
    counts = result.get_counts(0)
    qubit_counts = [marginal_counts(counts, [qubit]) for qubit in range(3*M)]
    ex_counts=marginal_counts(counts, indices=[int(3*M+np.mod(t,rate))])
    try:
        ex_q=(ex_counts['1']/numberOfShots)*3-tm
    except KeyError as error:
        ex_q=0
    #read post streaming probabilities
    fout = np.zeros((3,M))
    for k in range(M):
        if '1' in qubit_counts[2+3*k]:
            fout[0][k] = qubit_counts[2+3*k]['1'] / numberOfShots
        if '1' in qubit_counts[1+3*k]:
            fout[1][k] = qubit_counts[1+3*k]['1'] / numberOfShots
        if '1' in qubit_counts[0+3*k]:
            fout[2][k] = qubit_counts[0+3*k]['1'] / numberOfShots
    f = fout

    
    if np.mod(t,min(rate,short))==0 or t<=2*rate:
        f[0][M-1] = tm-f[2][M-1]-f[1][M-1]
        f[2][0] = bc-f[1][0]-f[0][0]

    else:
        f[0][M-1]=bcr[0]
        f[1][M-1]=bcr[1]
        f[2][M-1]=bcr[2]
        f[0][0]=bcl[0]        
        f[1][0]=bcl[1]        
        f[2][0]=bcl[2]        
    
    return f, qc,ex_q

# number operators
n1 = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]])

n2 = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]])

n3 = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]])


 
# collision operator - Diffusion equation
coeff = np.exp(-1j*pi/6)/sqrt(3)
a = np.exp(1j*pi/6)*sqrt(3)
b = np.exp(1j*2*pi/3)
C_Diffusion = coeff *  np.array([[a, 0, 0, 0, 0, 0, 0, 0],
                                  [0, b, 1, 1, 0, 0, 0, 0],
                                  [0, 1, b, 1, 0, 0, 0, 0],
                                  [0, 1, 1, b, 0, 0, 0, 0],
                                  [0, 0, 0, 0, b, 1, 1, 0],
                                  [0, 0, 0, 0, 1, b, 1, 0],
                                  [0, 0, 0, 0, 1, 1, b, 0],
                                  [0, 0, 0, 0, 0, 0, 0, a]])



def classicalOneTimeStep(f, state, M,t,rate):
#     initial combined state
    initial_state = np.zeros((M, 8))
    for i in range(M):
        initial_state[i][0] = np.sqrt( f[0][i] * f[1][i] * f[2][i] )
        initial_state[i][1] = np.sqrt( (1-f[2][i]) * f[0][i]  * f[1][i] )
        initial_state[i][2] = np.sqrt( (1-f[1][i]) * f[0][i]  * f[2][i] )
        initial_state[i][3] = np.sqrt( (1-f[0][i]) * f[2][i]  * f[1][i] )
        initial_state[i][4] = np.sqrt( (1-f[0][i]) * (1-f[1][i]) * f[2][i] )
        initial_state[i][5] = np.sqrt( (1-f[0][i]) * (1-f[2][i]) * f[1][i] )
        initial_state[i][6] = np.sqrt( (1-f[1][i]) * (1-f[2][i]) * f[0][i] )
        initial_state[i][7] = np.sqrt( (1-f[0][i]) * (1-f[1][i]) * (1-f[2][i]) )
        
#     post-collision state (8xlattice_sites)
    post_collision_state = C_Diffusion.dot(initial_state.conjugate().transpose())
    
   # post-collision distribution
    post_collision_distribution = np.zeros((3, M))
    for i in range(M):
        post_collision_distribution[0][i] = post_collision_state.conjugate().transpose()[i].dot(n1.dot( post_collision_state.transpose()[i]))
        post_collision_distribution[1][i] = post_collision_state.conjugate().transpose()[i].dot(n2.dot( post_collision_state.transpose()[i]))
        post_collision_distribution[2][i] = post_collision_state.conjugate().transpose()[i].dot(n3.dot( post_collision_state.transpose()[i]))

#     STREAMING STEP 
    f[0][1:M] = post_collision_distribution[0][0:M-1]
    f[1][0:M] = post_collision_distribution[1][0:M]
    f[2][0:M-1] = post_collision_distribution[2][1:M]
    f[0][0] = 1-f[2][0]-f[1][0]
    f[2][M-1] = 0
    for i in range(M):
            if np.sum(f,axis=0)[i]>tm:
                if state[i]<1:
                    state[i]+=(np.sum(f,axis=0)[i]-tm)*cp/la
                    column_sum = np.sum(f[:, i])
                    if column_sum != 0:  
                        f[:, i] *= tm / column_sum

                if state[i]>=1:
                    column_sum = np.sum(f[:, i])                    
                    f[0][i]+=f[0][i]*(state[i]-1)*la/cp/column_sum
                    f[1][i]+=f[1][i]*(state[i]-1)*la/cp/column_sum
                    f[2][i]+=f[2][i]*(state[i]-1)*la/cp/column_sum
                    state[i]=1
    
    return f, state



L = 16  # domain length 
M = L+1  # number of lattice sites (from 0 to L)
n = 3*M   # number of qubits
x = np.array(range(M)) # 1D lattice
rate=1

numberOfShots = 16384  # number of shots
maxT = 110   # number of time steps

#initialization
fini, rho,state = ini(x, w, cx, ux, csq)    

qc = QuantumCircuit(3*M)
qc = update_encoding(qc, fini, M)
fq_classical = fini
fq_quantum = fini
fClassical = fini
state_cc=np.copy(state)
state_qc=np.copy(state)
lam=fsolve(f,1)
gam=lam[0]

fq_quantum_sol=fini
fq_quantum_liq=fini
q_state=np.copy(state)
full_liq=0
ch_liq=1
ch_sol=0
full_sol=0
ex=0
rhoq_quantum = fq_quantum[0] + fq_quantum[1] + fq_quantum[2]

#points to track
point_liq = int(1) 
point_sol = int(2) 

def interpolate_x_tot(x_base_points, x_tot_points, x_base_query):

    f = interp1d(x_base_points, x_tot_points, 
                 kind='linear', 
                 bounds_error=False, 
                 fill_value=np.nan)
    
    return f(x_base_query)

# Store previous boundary values
left_boundary_values = [[], [], []]
mid_boundary_values = [[], [], []]
right_boundary_values = [[], [], []]
left_boundary_values_c = [[], [], []]
mid_boundary_values_c = [[], [], []]
right_boundary_values_c = [[], [], []]
short=rate+1
count=3
ch_old=0
ex_plus=0
ex_minus=0
liq_frac=state[0]
errs=[]

for t in range(maxT+1):
    print('t = ', t)
    #quantum implementation, classical streaming
    fq_classical, qc1, state_qc = oneTimeStep_classicalStreaming(fq_classical, M, state_qc, numberOfShots, Aer.get_backend(backend)) 
    rhoq_classical = fq_classical[0] + fq_classical[1] + fq_classical[2]
    
    #quantum implementation, quantum streaming
    if not any(0 < x < 1 for x in q_state[1:]): #no node currently melting

        fq_quantum, qc2, exc = oneTimeStep_quantumStreaming(fq_quantum, M, numberOfShots, Aer.get_backend(backend), t, maxT,1,fq_quantum[:,0]/rhoq_quantum[0],fq_quantum[:,-1])
        rhoq_quantum = fq_quantum[0] + fq_quantum[1] + fq_quantum[2]
    
    else:
        if np.mod(t,min(rate,short))==1 or t<2*rate or min(rate,short)==1:     #reinitialization step   
            fq_quantum_liq=np.copy(fq_quantum)
            fq_quantum_liq=fq_quantum_liq[:,:ch_liq+1]
            
            fq_quantum_sol=np.copy(fq_quantum)
            fq_quantum_sol=fq_quantum_sol[:,ch_liq:]
            liq_bcl=fq_quantum_liq[:,0]
            liq_bcr=fq_quantum_liq[:,-1]            
            sol_bcl=fq_quantum_sol[:,0]
            sol_bcr=fq_quantum_sol[:,-1]

        
        fq_quantum_liq, qc2,ex_term = oneTimeStep_quantumStreaming_liq(fq_quantum_liq, int(np.size(fq_quantum_liq)/3), numberOfShots, Aer.get_backend(backend), t, maxT,1,liq_bcl,liq_bcr)
        rhoq_quantum_liq = fq_quantum_liq[0] + fq_quantum_liq[1] + fq_quantum_liq[2]
        ex+=ex_term
        fq_quantum_sol, qc2,ex_term_sol = oneTimeStep_quantumStreaming(fq_quantum_sol, int(np.size(fq_quantum_sol)/3), numberOfShots, Aer.get_backend(backend), t, maxT,tm,sol_bcl,sol_bcr)
        rhoq_quantum_sol = fq_quantum_sol[0] + fq_quantum_sol[1] + fq_quantum_sol[2]
        ex+=ex_term_sol
        ex_plus+=ex_term
        ex_minus-=ex_term_sol
        
        if np.mod(t,min(rate,short))==0 or t<2*rate:  #reinitialization step
            if ch_liq!=0:
                for i in range(M):
                    if i<ch_liq:
                        fq_quantum[0][i]=fq_quantum_liq[0][i]
                        fq_quantum[1][i]=fq_quantum_liq[1][i]
                        fq_quantum[2][i]=fq_quantum_liq[2][i]
                    elif i>ch_liq:
                        fq_quantum[0][i]=fq_quantum_sol[0][i-ch_liq]
                        fq_quantum[1][i]=fq_quantum_sol[1][i-ch_liq]
                        fq_quantum[2][i]=fq_quantum_sol[2][i-ch_liq]
                    else:
                        fq_quantum[0][i]=fq_quantum_sol[0][i-ch_liq]
                        fq_quantum[1][i]=fq_quantum_sol[1][i-ch_liq]
                        fq_quantum[2][i]=fq_quantum_liq[2][i]
            rhoq_quantum = fq_quantum[0] + fq_quantum[1] + fq_quantum[2]
    full_liq=0
    ch_liq=0
    ch_sol=0
    full_sol=0
    for i in range(M):          #update liquid fraction
        if rhoq_quantum[i]>=tm:
            if q_state[i]<1:
                if not any(0 < x < 1 for x in q_state[1:]):
                    q_state[i]+=(rhoq_quantum[i]-tm)*cp/la
                q_state[i]+=(ex)*cp/la
                fq_quantum[0][i]*=tm/rhoq_quantum[i]
                fq_quantum[1][i]*=tm/rhoq_quantum[i]
                fq_quantum[2][i]*=tm/rhoq_quantum[i]
            if q_state[i]>=1:
                fq_quantum[0][i]+=fq_quantum[0][i]/rhoq_quantum[i]*(q_state[i]-1)*la/cp
                fq_quantum[1][i]+=fq_quantum[1][i]/rhoq_quantum[i]*(q_state[i]-1)*la/cp
                fq_quantum[2][i]+=fq_quantum[2][i]/rhoq_quantum[i]*(q_state[i]-1)*la/cp
                q_state[i]=1
        if q_state[i]==1:
            full_liq=i
            ch_liq=full_liq+1
    rhoq_quantum = fq_quantum[0] + fq_quantum[1] + fq_quantum[2]
    liq_frac=q_state[ch_liq]
    ex=0
    if t>5:     #single timestep reinitialization around node changes
        if ch_liq!=ch_old:
            q_series=[]
            ch_old=ch_liq
            print(fq_quantum_liq)
            print(fq_quantum_sol)
        q_series.append([t, q_state[ch_liq]])
    if short<rate and count>0:
        short=1
        count-=1
    elif count==0:
        short=rate+1
    if t>5 and len(q_series)>=5 and short==rate+1:
        t_val = [item[0] for item in q_series]  
        y_val = [item[1] for item in q_series]  
        slope, intercept, _, _, _ = linregress(t_val, y_val)
        t_shift = (tb - intercept) / slope
        if t+rate>t_shift-1:
            short=1
            count=np.floor(10+t_shift-t)    
        else:
            short=rate+1
    ex_minus=0
    ex_plus=0
    
    #classical implementation
    fClassical,state_cc = classicalOneTimeStep(fClassical, state_cc, M,t,rate)
    rhoClassical = fClassical[0] + fClassical[1] + fClassical[2]
    
    #analytical implementation
    loc=2*gam*np.sqrt(alpha*t*dt)
    key_num=int((M+1)/2)
    x_liq=np.zeros(int((M+1)/2))
    x_sol=np.zeros(key_num)
    x_base_liq=np.linspace(0,loc,int((M+1)/2))
    x_base_sol=np.linspace(loc,L,key_num)
    x_base=np.concatenate((x_base_liq,x_base_sol[1:]))
    n_terms=np.arange(1,200)
    for i in range(int((M+1)/2)):
        x_liq[i]=tb-(tb-tm)*(special.erf((i)*loc/((M+1)/2-1)/(2*np.sqrt(alpha*t*dt))))/special.erf(gam)
    for i in range(key_num):
        x_sol[i]=tm + (to - tm) * (x_base_sol[i]) / L + np.sum(
            (2 / (n_terms * np.pi)) * (to - tm) * np.sin(n_terms * np.pi * (x_base_sol[i]) / L) *
            np.exp(-alpha * (n_terms * np.pi / L)**2 * t)
        )
    x_tot=np.concatenate((x_liq,x_sol[1:]))
    
    if np.mod(t,rate)==0:
        # Interpolate the LBM values based on liquid fraction
        boun_q=ch_liq-0.5+q_state[ch_liq]
        boun_c=ch_liq-0.5+state_cc[ch_liq]
        
        if q_state[ch_liq]!=0:
            qq_interp = np.concatenate([
            np.linspace(0, boun_q, ch_liq + 1),
            np.linspace(boun_q + (M - ch_liq - 1) / (M - ch_liq),boun_q + (M - ch_liq - 1) / (M - ch_liq)+M-ch_liq-2, M - ch_liq - 1)
        ])  
        else:
            qq_interp=np.linspace(0,M-1,M)
        if state_cc[ch_liq]!=0:    
            cc_interp = np.concatenate([
                np.linspace(0, boun_c, ch_liq + 1),
                np.linspace(boun_c + (M - ch_liq - 1) / (M - ch_liq),boun_c + (M - ch_liq - 1) / (M - ch_liq)+M-ch_liq-2, M - ch_liq - 1)
            ])   
        
        else:
            cc_interp=np.linspace(0,M-1,M)
        
        try:
            q_tot_interp = [interpolate_x_tot(qq_interp, rhoq_quantum, point) for point in range(M)]
            c_tot_interp = [interpolate_x_tot(cc_interp, rhoClassical, point) for point in range(M)]
            MSE=math.sqrt(mean_squared_error(qq_interp,cc_interp))
            errs.append(MSE)
            
            plt.plot(rate*np.linspace(0,len(errs[4:])-1,len(errs[4:])) ,errs[4:])
            plt.ylabel('Error', fontsize='large')
            plt.xlabel('Timesteps', fontsize='large')
            plt.savefig('error_1.pdf', format='pdf', bbox_inches='tight', dpi=300)
            #plt.title('Error between quantum and analytical solutions')
            plt.show()
        except ValueError as e:
            e
        
        plt.plot(qq_interp, rhoq_quantum,'s', markersize = 6, markerfacecolor='none')
        plt.plot(cc_interp, rhoClassical, 'k')
        plt.plot(x_base,x_tot)
        plt.xlabel('Lattice site', fontsize='large')
        plt.ylabel('Temperature', fontsize='large')
        plt.legend(['Quantum', 'Classical', 'Analytic'], fontsize='large')
        plt.savefig('LvT_1.pdf', format='pdf', bbox_inches='tight', dpi=300)
        plt.show()
        
        
        # Define colors for each type
        quantum_color = 'blue'
        classical_color = 'red'
        analytical_color = 'green'
        
        # Plot with same colors but different line styles
        #plt.plot(time_series, temperature_at_point_liq, color=quantum_color, linestyle='-', label='Quantum')
        #plt.plot(time_series, temperature_at_point_sol, color=quantum_color, linestyle='--')
        
        #plt.plot(time_series, temperature_at_point_liq_c, color=classical_color, linestyle='-', label='Classical')
        #plt.plot(time_series, temperature_at_point_sol_c, color=classical_color, linestyle='--')
        
        #plt.plot(time_series, temperature_at_point_liq_a, color=analytical_color, linestyle='-', label='Analytical')
        #plt.plot(time_series, temperature_at_point_sol_a, color=analytical_color, linestyle='--')
        
        #plt.xlabel('Time', fontsize='large')
        #plt.ylabel('Temperature',fontsize='large')
        #plt.legend()
        #plt.savefig('TvT_1.pdf', format='pdf', bbox_inches='tight', dpi=300)
        #plt.show()
        
print('Done')


MSE = mean_squared_error(rhoClassical, rhoq_classical)
RMSE = math.sqrt(MSE)
print("RMSE of quantum solution with classical streaming:\n", RMSE)

MSE = mean_squared_error(rhoClassical, rhoq_quantum)
RMSE = math.sqrt(MSE)
print("RMSE of quantum solution with quantum streaming:\n", RMSE)
