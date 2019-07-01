#!/usr/bin/env python
# encoding: utf-8
r"""
Riemann solvers for the shallow water equations.

The available solvers are:
 * Roe - Use Roe averages to caluclate the solution to the Riemann problem
 * HLL - Use a HLL solver
 * Exact - Use a newton iteration to calculate the exact solution to the
        Riemann problem

.. math::
    q_t + f(q)_x = 0

where

.. math::
    q(x,t) = \left [ \begin{array}{c} h \\ h u \end{array} \right ],

the flux function is

.. math::
    f(q) = \left [ \begin{array}{c} h u \\ hu^2 + 1/2 g h^2 \end{array}\right ].

and :math:`h` is the water column height, :math:`u` the velocity and :math:`g`
is the gravitational acceleration.

:Authors:
    Kyle T. Mandli (2009-02-05): Initial version
"""
# ============================================================================
#      Copyright (C) 2009 Kyle T. Mandli <mandli@amath.washington.edu>
#
#  Distributed under the terms of the Berkeley Software Distribution (BSD)
#  license
#                     http://www.opensource.org/licenses/
# ============================================================================

import numpy as np

num_eqn = 2
num_waves = 2


## this function determines the solution type of 1d SWE, whether you have one shock, one rarefaction, two shocks, two rarefactions, etc
def riemanntype(hL, hR, uL, uR, maxiter, drytol, g):
    h_min = min(hR,hL)
    h_max = max(hR,hL)
    delu = uR - uL

    if (h_min <= drytol):
        hm = 0.0
        um = 0.0
        s1m = uR + uL - 2.0 * np.sqrt(g * hR) + 2.0 * np.sqrt(g * hL)  # since uR or uL will be zero and so will hL or hR
        s2m = uR + uL - 2.0 * np.sqrt(g * hR) + 2.0 * np.sqrt(g * hL)
        if (hL <= 0.0):
            rare2 = True   # since hR will be nonzero, the rarefaction happens with the 2-wave
            rare1 = False
        else:
            rare1 = True
            rare2 = False
    else:
        F_min = delu + 2.0 * (np.sqrt(g * h_min) - np.sqrt(g * h_max))
        F_max = delu + (h_max - h_min) * (np.sqrt(0.5 * g * (h_max + h_min) / (h_max * h_min)))

        if (F_min > 0.0): #2-rarefactions
            hm = (1.0 / (16.0 * g)) * max(0.0, - delu + 2.0 * (np.sqrt(g * hL) + np.sqrt(g * hR)))**2
            um = uL + 2.0 * (np.sqrt(g * hL) - np.sqrt(g * hm))
            s1m = uL + 2.0 * np.sqrt(g * hL) - 3.0 * np.sqrt(g * hm)
            s2m = uR - 2.0 * np.sqrt(g * hR) + 3.0 * np.sqrt(g * hm)
            rare1 = True
            rare2 = True

        elif (F_max <= 0.0): # !2 shocks
            # root finding using a Newton iteration on sqrt(h)===
            h0 = h_max
            for iter in range(maxiter):
                gL = np.sqrt(0.5 * g * (1 / h0 + 1 / hL))
                gR = np.sqrt(0.5 * g * (1 / h0 + 1 / hR))
                F0 = delu + (h0 - hL) * gL + (h0 - hR) * gR
                dfdh = gL - g * (h0 - hL) / (4.0 * (h0**2) * gL) + gR - g * (h0 - hR) / (4.0 * (h0**2) * gR)
                slope = 2.0 * np.sqrt(h0) * dfdh
                h0 = (np.sqrt(h0) - F0 / slope)**2

            hm = h0
            u1m = uL - (hm-hL) * np.sqrt((0.5 * g) * (1 / hm + 1 / hL))
            u2m = uR + (hm - hR) * np.sqrt((0.5 * g) * (1 / hm + 1 / hR))
            um = 0.5 * (u1m + u2m)
            s1m = u1m - np.sqrt(g * hm)
            s2m = u2m + np.sqrt(g * hm)
            rare1 = False
            rare2 = False

        else: #one shock one rarefaction
            h0 = h_min
            for iter in range(maxiter):
                F0 = delu + 2.0 * (np.sqrt(g * h0) - np.sqrt(g * h_max)) + (h0 - h_min) * np.sqrt(0.5 * g * (1 / h0 + 1 / h_min))
                slope = (F_max - F0) / (h_max - h_min)
                h0 = h0 - F0 / slope

            hm = h0
            if (hL > hR):
                um = uL + 2.0 * np.sqrt(g * hL) - 2.0 * np.sqrt(g * hm)
                s1m = uL + 2.0 * np.sqrt(g * hL) - 3.0 * np.sqrt(g * hm)
                s2m = uL + 2.0 * np.sqrt(g * hL) - np.sqrt(g * hm)
                rare1 = True
                rare2 = False
            else:
                s2m = uR - 2.0 * np.sqrt(g * hR) + 3.0 * np.sqrt(g * hm)
                s1m = uR - 2.0 * np.sqrt(g * hR) + np.sqrt(g * hm)
                um = uR - 2.0 * np.sqrt(g * hR) + 2.0 * np.sqrt(g * hm)
                rare2 = True
                rare1 = False

    return hm, s1m, s2m, rare1, rare2


def shallow_fwave_1d(q_l, q_r, aux_l, aux_r, problem_data):
    r"""Shallow water Riemann solver using fwaves

    Also includes support for bathymetry but be wary if you think you might have
    dry states as this has not been tested.

    *problem_data* should contain:
     - *grav* - (float) Gravitational constant
     - *sea_level* - (float) Datum from which the dry-state is calculated.

    :Version: 1.0 (2014-09-05)
    """

    g = problem_data['grav']

    num_rp = np.size(q_l,1)
    num_eqn = 2
    num_waves = 2

    # Output arrays
    fwave = np.zeros( (num_eqn, num_waves, num_rp) )
    s = np.zeros( (num_waves, num_rp) )
    amdq = np.zeros( (num_eqn, num_rp) )
    apdq = np.zeros( (num_eqn, num_rp) )

    # Extract state
    u_l = np.where(q_l[0,:] - problem_data['sea_level'] > 1e-3,
                   q_l[1,:] / q_l[0,:], 0.0)
    u_r = np.where(q_r[0,:] - problem_data['sea_level'] > 1e-3,
                   q_r[1,:] / q_r[0,:], 0.0)
    phi_l = q_l[0,:] * u_l**2 + 0.5 * g * q_l[0,:]**2
    phi_r = q_r[0,:] * u_r**2 + 0.5 * g * q_r[0,:]**2

    # Speeds
    s[0,:] = u_l - np.sqrt(g * q_l[0,:])
    s[1,:] = u_r + np.sqrt(g * q_r[0,:])

    delta1 = q_r[1,:] - q_l[1,:]
    delta2 = phi_r - phi_l + g * 0.5 * (q_r[0,:] + q_l[0,:]) * (aux_r[0,:] - aux_l[0,:])

    beta1 = (s[1,:] * delta1 - delta2) / (s[1,:] - s[0,:])
    beta2 = (delta2 - s[0,:] * delta1) / (s[1,:] - s[0,:])

    fwave[0,0,:] = beta1
    fwave[1,0,:] = beta1 * s[0,:]
    fwave[0,1,:] = beta2
    fwave[1,1,:] = beta2 * s[1,:]

    for m in range(num_eqn):
        for mw in range(num_waves):
            amdq[m,:] += (s[mw,:] < 0.0) * fwave[m,mw,:]
            apdq[m,:] += (s[mw,:] >= 0.0) * fwave[m,mw,:]

    return fwave, s, amdq, apdq


# gets the fwave from one RP
def riemann_fwave_1d(hL, hR, huL, huR, bL, bR, uL, uR, phiL, phiR, s1, s2, g):
    num_eqn = 2
    num_waves = 2
    fw = np.zeros((num_eqn, num_waves))

    delh = hR - hL
    delhu = huR - huL
    delb = bR - bL
    delphidecomp = phiR - phiL + g * 0.5 * (hL + hR) * delb

    beta1 = (s2 * delhu - delphidecomp) / (s2 - s1)
    beta2 = (delphidecomp - s1 * delhu) / (s2 - s1)

    # 1st nonlinear wave
    fw[0,0] = beta1
    fw[1,0] = beta1 * s1

    # 2nd nonlinear wave
    fw[0,1] = beta2
    fw[1,1] = beta2 * s2

    return fw

# determines whether water will overtop a barrier from left to right and right to left
def barrier_passing(hL, hR, huL, huR, bL, bR, wall_height, drytol, g, maxiter):

    L2R = False
    R2L = False
    hstarL = 0.0
    hstarR = 0.0

    if (hL > drytol):
        uL = huL / hL
        hstar,_,_,_,_ = riemanntype(hL, hL, uL, -uL, maxiter, drytol, g)
        hstartest = max(hL, hstar)
        if (hstartest + bL > 0.5*(bL+bR)+wall_height): # why [0.5(bL+bR) + wall_height]?
            L2R = True
            hstarL = hstartest + bL - 0.5*(bL+bR) - wall_height

    if (hR > drytol):
        uR = huR / hR
        hstar,_,_,_,_ = riemanntype(hR, hR, -uR, uR, maxiter, drytol, g)
        hstartest = max(hR, hstar)
        if (hstartest + bR > 0.5*(bL+bR)+wall_height):
            R2L = True
            hstarR = hstartest + bR - 0.5*(bL+bR) - wall_height

    return L2R, R2L, hstarL, hstarR

# in case the wall is set on edge, what will be the fluctuations is calculated by this function
def redistribute_fwave(q_l, q_r, aux_l, aux_r, wall_height, drytol, g, maxiter):

    fwave = np.zeros((2, 2, 2))
    s = np.zeros((2, 2))
    amdq = np.zeros((2, 2))
    apdq = np.zeros((2, 2))

    q_wall = np.zeros((2,3))
    aux_wall = np.zeros((1,3))
    s_wall = np.zeros((2,1))
    gamma = np.zeros((2,2))
    amdq_wall = np.zeros((2,1))
    apdq_wall = np.zeros((2,1))

    # hbox method
    q_wall[:,[0]] = q_l.copy() # so q_l,r have to be 2x1 arrays--is that true?
    q_wall[:,[2]] = q_r.copy() # what about q_wall[:,1]?
    # print ("aux_l.shape: ", aux_l.shape)

    aux_wall[0,0] = 0 #aux_l.copy()
    aux_wall[0,2] = 0 #aux_r.copy()
    aux_wall[0,1] = 0.5*(aux_wall[0,0] + aux_wall[0,2]) + wall_height # assume that the wall is on top of bathy

    L2R, R2L, hstarL, hstarR = barrier_passing(q_wall[0,0], q_wall[0,2], q_wall[1,0], q_wall[1,2], aux_wall[0,0], aux_wall[0,2], wall_height, drytol, g, maxiter)
    if (L2R==True or R2L==True): # what about other cases?
        q_wall[0,1] = 0.5*(hstarL+hstarR)  # why these?
        q_wall[1,1] = q_wall[0,1]  * (q_wall[1,0] + q_wall[1,2])/(q_wall[0,0] + q_wall[0,2])

    q_wall_l = q_wall[:,:-1]
    q_wall_r = q_wall[:,1:]
    aux_wall_l = aux_wall[:,:-1]
    aux_wall_r = aux_wall[:,1:]

    for i in range(2):  # i stands for the two ghost problems
        hL = q_wall_l[0,i]
        hR = q_wall_r[0,i]
        huL = q_wall_l[1,i]
        huR = q_wall_r[1,i]
        bL = aux_wall_l[0,i]
        bR = aux_wall_r[0,i]

        # Check wet/dry states
        if (hR > drytol): # right state is not dry
            uR = huR / hR
            phiR = 0.5 * g * hR**2 + huR**2 / hR
        else:
            hR = 0.0
            huR = 0.0
            uR = 0.0
            phiR = 0.0

        if (hL > drytol):
            uL = huL / hL
            phiL = 0.5 * g * hL**2 + huL**2 / hL
        else:
            hL = 0.0
            huL = 0.0
            uL = 0.0
            phiL = 0.0

        if (hL > drytol or hR > drytol):
            wall = np.ones(2)
            if (hR <= drytol):
                hstar,_,_,_,_ = riemanntype(hL, hL, uL, -uL, maxiter, drytol, g)
                hstartest = max(hL, hstar)
                if (hstartest + bL <= bR):
                    wall[1] = 0.0 # why wall[1] and not wall[0]?
                    hR = hL
                    huR = -huL
                    bR = bL
                    phiR = phiL
                    uR = -uL
                elif (hL + bL <= bR):
                    bR = hL + bL

            if (hL <= drytol):
                hstar,_,_,_,_ = riemanntype(hR, hR, -uR, uR, maxiter, drytol, g)
                hstartest = max(hR, hstar)
                if (hstartest + bR <= bL):
                    wall[0] = 0.0
                    hL = hR
                    huL = -huR
                    bL = bR
                    phiL = phiR
                    uL = -uR
                elif (hR + bR <= bL):
                    bL = hR + bR

            sL = uL - np.sqrt(g * hL)
            sR = uR + np.sqrt(g * hR)
            uhat = (np.sqrt(g * hL) * uL + np.sqrt(g * hR) * uR) / (np.sqrt(g * hR) + np.sqrt(g * hL))
            chat = np.sqrt(g * 0.5 * (hR + hL))
            sRoe1 = uhat - chat
            sRoe2 = uhat + chat
            s1 = min(sL, sRoe1)
            s2 = max(sR, sRoe2)
            fw = riemann_fwave_1d(hL, hR, huL, huR, bL, bR, uL, uR, phiL, phiR, s1, s2, g)
            # print ("wall: ", wall[:])
            # print("s1: ", s1)
            # print("sR: ", sR)
            # print("sRoe2: ", sRoe2)
            # print("s2: ", s2)
            s[0,i] = s1 * wall[0]
            s[1,i] = s2 * wall[1]
            fwave[:,0,i] = fw[:,0] * wall[0]
            fwave[:,1,i] = fw[:,1] * wall[1]
            # print("fw: ", fw)

            for mw in range(num_waves):
                if (s[mw,i] < 0):
                    amdq[:,i] += fwave[:,mw,i]
                elif (s[mw,i] > 0):
                    apdq[:,i] += fwave[:,mw,i]
                # else:
                #     amdq[:,i] += 0.5 * fwave[:,mw,i]
                #     apdq[:,i] += 0.5 * fwave[:,mw,i]


    s_wall[0] = np.min(s[:])
    s_wall[1] = np.max(s[:])

    # s_wall[0] = s[0,0]
    # s_wall[1] = s[1,1]

    if s_wall[1] - s_wall[0] != 0.0:
        gamma[0,0] = (s_wall[1] * np.sum(fwave[0,:,:]) - np.sum(fwave[1,:,:])) / (s_wall[1] - s_wall[0])
        gamma[0,1] = (np.sum(fwave[1,:,:]) - s_wall[0] * np.sum(fwave[0,:,:])) / (s_wall[1] - s_wall[0])
        gamma[1,0] = gamma[0,0] * s_wall[0]
        gamma[1,1] = gamma[0,1] * s_wall[1]

    wave_wall = gamma
    # print("gamma[0,:]: ", gamma[0,:])
    for mw in range(2):
        if (s_wall[mw] < 0):
            amdq_wall[:] += gamma[:,mw]
        elif (s_wall[mw] > 0):
            apdq_wall[:] += gamma[:,mw]

    # print("amdq_wall: ", amdq_wall)
    # print("apdq_wall: ", apdq_wall)

    return wave_wall, s_wall, amdq_wall, apdq_wall


# using/calling as main solver #
# solves SWE 1D with fwaves, takes care of dry case, and when near the barrier edge, calls the specific solver
def shallow_fwave_dry_1d(q_l, q_r, aux_l, aux_r, problem_data):
    # print("shallow_fwave_hbox_dry_1d")
    print(q_l)
    print(q_r)
    g = 9.8# problem_data['grav']
    drytol = 0.001#problem_data['dry_tolerance']
    maxiter = 1# problem_data['max_iteration']
    num_rp_sh = q_l.shape
    print(num_rp_sh)
    num_rp = int(num_rp_sh[1])
    num_eqn = 2
    num_waves = 2
    num_ghost = 2

    # location of barrier
    nw = problem_data['wall_position']

    # Output arrays
    fwave = np.zeros((num_eqn, num_waves, num_rp))
    s = np.zeros((num_waves, num_rp))
    amdq = np.zeros((num_eqn, num_rp))
    apdq = np.zeros((num_eqn, num_rp))

    for i in range(num_rp):
        hL = q_l[0,i]
        hR = q_r[0,i]
        huL = q_l[1,i]
        huR = q_r[1,i]
        bL = aux_l[0,i]
        bR = aux_r[0,i]

        # Check wet/dry states
        if (hR > drytol): # right state is not dry
            uR = huR / hR
            phiR = 0.5 * g * hR**2 + huR**2 / hR
        else:
            hR = 0.0
            huR = 0.0
            uR = 0.0
            phiR = 0.0

        if (hL > drytol):
            uL = huL / hL
            phiL = 0.5 * g * hL**2 + huL**2 / hL
        else:
            hL = 0.0
            huL = 0.0
            uL = 0.0
            phiL = 0.0
        ### when you get to the barrier containing cell
        if i == nw:
            qLL = q_r[:,[i-1]]
            qRR = q_r[:,[i+2]]
            qL = q_l[:,[i]]
            qR = q_r[:,[i]]

            s_wall,s2,s3,amdq_small_l,apdq_small_l,apdq_small_r,amdq_small_r,amdq_LL,apdq_RR = barrier_solver(qLL,qL,qR,qRR,0.001,1.75)
            amdq[:,i] = amdq[:,i] + amdq_small_l.reshape(amdq[:,i].shape)
            apdq[:,i] = apdq[:,i] + apdq_small_l.reshape(apdq[:,i].shape)
            amdq[:,i+1] = amdq[:,i+1] + amdq_small_r.reshape(amdq[:,i].shape)
            apdq[:,i+1] = apdq[:,i+1] + apdq_small_r.reshape(apdq[:,i].shape)
            amdq[:,i-1] = amdq[:,i-1] + amdq_LL.reshape(amdq[:,i].shape)
            apdq[:,i+2] = apdq[:,i+2] + apdq_RR.reshape(apdq[:,i].shape)

            i += 2
           ### since you got the fluctuations for both i=loc_bar (sub_normal sized qL of barrier containing cell) and loc_bar+1 (sub_normal sized qR of barrier containing cell), jump to i+2, the normal sized cell right to the barrier-containing cell

        if (hL > drytol or hR > drytol):
            wall = np.ones(2)
            if (hR <= drytol):
                hstar,_,_,_,_ = riemanntype(hL, hL, uL, -uL, maxiter, drytol, g)
                hstartest = max(hL, hstar)
                if (hstartest + bL <= bR):
                    wall[1] = 0.0
                    hR = hL
                    huR = -huL
                    bR = bL
                    phiR = phiL
                    uR = -uL
                elif (hL + bL <= bR):
                    bR = hL + bL

            if (hL <= drytol):
                hstar,_,_,_,_ = riemanntype(hR, hR, -uR, uR, maxiter, drytol, g)
                hstartest = max(hR, hstar)
                if (hstartest + bR <= bL):
                    wall[0] = 0.0
                    hL = hR
                    huL = -huR
                    bL = bR
                    phiL = phiR
                    uL = -uR
                elif (hR+ bR <= bL):
                    bL = hR + bR

            sL = uL - np.sqrt(g * hL)
            sR = uR + np.sqrt(g * hR)
            uhat = (np.sqrt(g * hL) * uL + np.sqrt(g * hR) * uR) / (np.sqrt(g * hR) + np.sqrt(g * hL))
            chat = np.sqrt(g * 0.5 * (hR + hL))
            sRoe1 = uhat - chat
            sRoe2 = uhat + chat
            s1 = min(sL, sRoe1)
            s2 = max(sR, sRoe2)
            fw = riemann_fwave_1d(hL, hR, huL, huR, bL, bR, uL, uR, phiL, phiR, s1, s2, g)

            s[0,i] = s1 * wall[0]
            s[1,i] = s2 * wall[1]
            fwave[:,0,i] = fw[:,0] * wall[0]
            fwave[:,1,i] = fw[:,1] * wall[1]

            for mw in range(num_waves):
                if (s[mw,i] < 0):
                    amdq[:,i] += fwave[:,mw,i]
                elif (s[mw,i] > 0):
                    apdq[:,i] += fwave[:,mw,i]
                else:
                    amdq[:,i] += 0.5 * fwave[:,mw,i]
                    apdq[:,i] += 0.5 * fwave[:,mw,i]

#    if problem_data['zero_width'] == True:
        # print("zero_width wall")
#        nw = problem_data['wall_position']
#        wall_height = problem_data['wall_height']
#        iw = nw + num_ghost - 1
        # print ("aux_l[iw:iw+2].shape: ", aux_l[0,iw:iw+2])
#        fwave[:,:,iw], s[:,iw], amdq[:,iw], apdq[:,iw] = redistribute_fwave(q_l[:,iw:iw+1].copy(), q_r[:,iw:iw+1].copy(), aux_l[0,iw:iw+1].copy(), aux_r[0,iw:iw+1].copy(), wall_height, drytol, g, maxiter)



    # f_handle = file('amdq_redist', 'a')
    # np.savetxt(f_handle, amdq)
    # f_handle.close()

    # f_handle = file('apdq_redist', 'a')
    # np.savetxt(f_handle, apdq)
    # f_handle.close()

    return fwave, s, amdq, apdq


## in the case that bathymetry = 0
## solves rieman problems with barrier located within a cell

def barrier_solver(qLL, qL, qR, qRR, alpha, barrier_height):
    #input:
    """qL = the left part of cell resulting from the dividing barrier
       qLL = the left cell of the barrier containining cell
       qR = the right part of cell resulting from the barrier
       qRR = the right cell of the barrier containing cell
       alpha = the portion of the left cell in the barrier cell
       barrier_height = the height of the barrier"""
    #output:
    """(1) fluxes that update the barrier cell
       (2) apdq fluxes that update the right-lying cell of the barrier cell
       (3) amdq fluxes that update the left-lying cell of the barrier cell"""
    problem_data={}
    problem_data['grav']=9.8
    problem_data['sea_level']=0.0
    problem_data['wall_position'] = 16
    problem_data['wall_height'] = 1.75
    problem_data['dry_tolerance'] = 0.001
    problem_data['max_iteration'] = 1
    problem_data['fraction'] = 0.001
    problem_data['num_rp'] = 32
    alpha = problem_data['fraction']

    # print qL and qR
    print('qL for barrier',qL)
    print('qR for barrier',qR)

   # Riemann subproblem 1:
    wave_wall, s_wall, amdq_wall, apdq_wall  = redistribute_fwave(qL+(1-alpha)*qLL, qR+(alpha)*qRR,np.zeros(qL.shape),np.zeros(qL.shape),barrier_height,0.001,9.8,50)

   # Riemann subproblem 2:
    fwave2, s2, amdq2, apdq2 = shallow_fwave_dry_1d(qLL, (1-alpha)*qR + (alpha)*qL,np.zeros(qL.shape),np.zeros(qL.shape),problem_data)

   # Riemann subproblem 3:
    fwave3, s3, amdq3, apdq3 = shallow_fwave_dry_1d((1-alpha)*qR + (alpha)*qL, qRR,np.zeros(qL.shape),np.zeros(qL.shape),problem_data)

    # whether overtops the barrier or not:
     #from left to right
    # Hbox values to left and right of barrier
    LST = qL+(1-alpha)*qLL
    RST = qR+alpha*qRR
    if LST[0] < 0.001:
        hstar,_,_,_,_ = riemanntype(0,0,0,0,1,0.001,9.8)
    else:
        hstar,_,_,_,_ = riemanntype(LST[0], LST[0], LST[1]/LST[0], -LST[1]/LST[0],1,.001,9.8)
    if hstar < 0.001:
        limiting_constant = 0
    else:
        limiting_constant = max(0, 1-(barrier_height/hstar))
     # from right to left
    if RST[0] < 0.001:
        hstar2,_,_,_,_ = riemanntype(0,0,0,0,1,0.001,9.8)
    else:
        hstar2,_,_,_,_ = riemanntype(RST[0],RST[0],-RST[1]/RST[0], RST[1]/RST[0], 1,0.001,9.8)
    if hstar2 < 0.001:
        limiting_constant2 = 0
    else:
        limiting_constant2 =max(0, 1 - (barrier_height/hstar2))

   # fwave for barrier cell:
    # the amdq for left part of divided cell:
    amdq_small_l = 0.9*amdq_wall + (limiting_constant2)*amdq3
    print('amdq_wall and amdq3', amdq_wall,amdq3)
    # the apdq for left part of divided cell:
    apdq_small_l= (1-limiting_constant)*apdq2

    # the apdq for right part of divided cell:
    apdq_small_r = 0.9*apdq_wall + (limiting_constant)*apdq2
    # the amdq for right part of divided cell:
    amdq_small_r = (1-limiting_constant2)*amdq3

   # amdq fwave for qLL:
    amdq_L = amdq2 + 0.1*amdq_wall

   # apdq fwave for qRR:
    apdq_R = 0.1*apdq_wall+apdq3

    return s_wall, s2, s3, amdq_small_l, apdq_small_l, apdq_small_r, amdq_small_r, amdq_L, apdq_R





## not using ##
def shallow_fwave_hbox_dry_1d(q_l, q_r, aux_l, aux_r, problem_data):
    # print("shallow_fwave_hbox_dry_1d")
    g = problem_data['grav']
    nw = problem_data['wall_position']
    wall_height = problem_data['wall_height']
    drytol = problem_data['dry_tolerance']
    maxiter = problem_data['max_iteration']
    alpha = problem_data['fraction']


    if problem_data['arrival_state'] == False: # what is arrival state? when wall is on an edge
        # print("arrival_state is False")
        num_rp = q_l.shape[1]
        num_eqn = 2
        num_waves = 2
        num_ghost = 2
        iw = nw + num_ghost - 1

        # Output arrays
        fwave = np.zeros((num_eqn, num_waves, num_rp))
        s = np.zeros((num_waves, num_rp))
        amdq = np.zeros((num_eqn, num_rp))
        apdq = np.zeros((num_eqn, num_rp))

        q_hbox = np.zeros((2,2))
        aux_hbox = np.zeros((1,2))

        ratio1 = 2.0 * alpha / (1 + alpha)
        ratio2 = 2.0 * (1 - alpha) / (2 - alpha)

        q_hbox[:,0] = ratio1 * q_r[:,iw-1] + (1 - ratio1) * q_r[:,iw-2]
        aux_hbox[0,0] = ratio1 * aux_r[0,iw-1] + (1 - ratio1) * aux_r[0,iw-2]
        q_hbox[:,1] = ratio2 * q_l[:,iw+1] + (1 - ratio2) * q_l[:,iw+2]
        aux_hbox[0,1] = ratio2 * aux_l[0,iw+1] + (1 - ratio2) * aux_l[0,iw+2]

        q_l[:,iw] = q_hbox[:,0]
        q_r[:,iw] = q_hbox[:,1]
        aux_l[0,iw] = aux_hbox[0,0]
        aux_r[0,iw] = aux_hbox[0,1]

        for i in range(num_rp):
            hL = q_l[0,i]
            hR = q_r[0,i]
            huL = q_l[1,i]
            huR = q_r[1,i]
            bL = aux_l[0,i]
            bR = aux_r[0,i]

            # Check wet/dry states
            if (hR > drytol): # right state is not dry
                uR = huR / hR
                phiR = 0.5 * g * hR**2 + huR**2 / hR
            else:
                hR = 0.0
                huR = 0.0
                uR = 0.0
                phiR = 0.0

            if (hL > drytol):
                uL = huL / hL
                phiL = 0.5 * g * hL**2 + huL**2 / hL
            else:
                hL = 0.0
                huL = 0.0
                uL = 0.0
                phiL = 0.0

            if (hL > drytol or hR > drytol):
                wall = np.ones(2)
                if (hR <= drytol):
                    hstar,_,_,_,_ = riemanntype(hL, hL, uL, -uL, maxiter, drytol, g)
                    hstartest = max(hL, hstar)
                    if (hstartest + bL <= bR):
                        wall[1] = 0.0
                        hR = hL
                        huR = -huL
                        bR = bL
                        phiR = phiL
                        uR = -uL
                    elif (hL + bL <= bR):
                        bR = hL + bL

                if (hL <= drytol):
                    hstar,_,_,_,_ = riemanntype(hR, hR, -uR, uR, maxiter, drytol, g)
                    hstartest = max(hR, hstar)
                    if (hstartest + bR <= bL):
                        wall[0] = 0.0
                        hL = hR
                        huL = -huR
                        bL = bR
                        phiL = phiR
                        uL = -uR
                    elif (hR + bR <= bL):
                        bL = hR + bR

                sL = uL - np.sqrt(g * hL)
                sR = uR + np.sqrt(g * hR)
                uhat = (np.sqrt(g * hL) * uL + np.sqrt(g * hR) * uR) / (np.sqrt(g * hR) + np.sqrt(g * hL))
                chat = np.sqrt(g * 0.5 * (hR + hL))
                sRoe1 = uhat - chat
                sRoe2 = uhat + chat
                s1 = min(sL, sRoe1)
                s2 = max(sR, sRoe2)
                fw = riemann_fwave_1d(hL, hR, huL, huR, bL, bR, uL, uR, phiL, phiR, s1, s2, g)

                s[0,i] = s1 * wall[0]
                s[1,i] = s2 * wall[1]
                fwave[:,0,i] = fw[:,0] * wall[0]
                fwave[:,1,i] = fw[:,1] * wall[1]

                for mw in range(num_waves):
                    if (s[mw,i] < 0):
                        amdq[:,i] += fwave[:,mw,i]
                    elif (s[mw,i] > 0):
                        apdq[:,i] += fwave[:,mw,i]
                    else:
                        amdq[:,i] += 0.5 * fwave[:,mw,i]
                        apdq[:,i] += 0.5 * fwave[:,mw,i]

        fwave[:,:,iw], s[:,iw], amdq[:,iw], apdq[:,iw] = redistribute_fwave(q_l[:,iw].copy(), q_r[:,iw].copy(), aux_l[0,iw].copy(), aux_r[0,iw].copy(), wall_height, drytol, g, maxiter)

        return fwave, s, amdq, apdq#, q_hbox, aux_hbox

    if problem_data['arrival_state'] == True:
        # print("arrival_state is True")
        return redistribute_fwave(q_l, q_r, aux_l, aux_r, wall_height, drytol, g, maxiter)










# augmented solver
def riemann_aug_JCP(maxiter,hL,hR,huL,huR,bL,bR,uL,uR,phiL,phiR,sE1,sE2,drytol,g):
    #Optimization: scipy for netwon method, and solve the matrix
    num_eqn = 2
    num_waves = 3 # the wave number here probably different with main function

    delh = hR - hL
    delhu = huR - huL
    delphi = phiR - phiL
    delb = bR - bL
    delnorm = delh**2 + delphi**2

    hstar, s1m, s2m, rare1, rare2 = riemanntype(hL,hR,uL,uR,maxiter,drytol,g)

    sw = np.zeros(num_waves)
    fw = np.zeros( (num_eqn, num_waves) )
    r = np.zeros((num_waves,num_waves))
    A = np.zeros((num_waves,num_waves))
    lamda = np.zeros(num_waves)
    beta = np.zeros(num_waves)
    dell = np.zeros(num_waves)

    lamda[0] = min(sE1,s2m)  # Modified Einfeldt speed
    lamda[2] = max(sE2,s1m)  # Modified Einfeldt speed
    sE1 = lamda[0]
    sE2 = lamda[2]
    lamda[1] = 0.0
    hstarHLL = max((huL-huR+sE2*hR-sE1*hL)/(sE2-sE1), 0.0)

    #determine the middle entropy corrector wave
    rarecorrectortest = False
    rarecorrector = False
    if (rarecorrectortest):
        sdelta = lamda[2] - lamda[0]
        raremin = 0.5
        raremax = 0.9
        if (rare1 and sE1*s1m < 0.0): raremin = 0.2
        if (rare2 and sE2*s2m < 0.0): raremin = 0.2
        if (rare1 or rare2):
            # see which rearefaction is larger
            rare1st = 3.0*(sqrt(g*hL)-sqrt(g*hm))
            rare2st = 3.0*(sqrt(g*hR)-sqrt(g*hm))
            if (max(rare1st,rare2st) > raremin*sdelta and max(rare1st,rare2st) < raremax*sdelta):
                rarecorrector = True
                if (rare1st > rare2st):
                    lamda[1]=s1m
                elif (rare2st > rare1st):
                    lamda[1]=s2m
                else:
                    lamda[1]=0.5*(s1m+s2m)

        if (hstarHLL < min(hL,hR)/5): rarecorrector=False # why?

    ## Is this correct 2-wave when rarecorrector == True ??
    for mw in range(num_waves):
        r[0,mw] = 1.0
        r[1,mw] = lamda[mw]
        r[2,mw] = (lamda[mw])**2

    if (not rarecorrector):
        lamda[1] = 0.5 * (lamda[0] + lamda[2])
    #   lamda(2) = max(min(0.5d0*(s1m+s2m),sE2),sE1)
        r[0,1] = 0.0
        r[1,1] = 0.0
        r[2,1] = 1.0

    # determin the steady state wave
    criticaltol = max(drytol*g, 1e-6)
    criticaltol_2 = np.sqrt(criticaltol)
    deldelh = -delb
    deldelphi = - 0.5 * g * (hR + hL) * delb

    # determne a few quanities needed for steady state wave if iterated
    hLstar = hL
    hRstar = hR
    uLstar = uL
    uRstar = uR
    huLstar = uLstar * hLstar
    huRstar = uRstar * hRstar

    # iterate to better determine the steady state wave
    convergencetol = 1e-6
    for iter in range(maxiter):
        #determine steady state wave (this will be subtracted from the delta vectors)
        if (min(hLstar,hRstar) < drytol and rarecorrector):
            rarecorrector = False
            hLstar = hL
            hRstar = hR
            uLstar = uL
            uRstar = uR
            huLstar = uLstar * hLstar
            huRstar = uRstar * hRstar
            lamda[2] = 0.5 * (lamda[1] + lamda[3])
            # lamda[2] = max(min(0.5*(s1m+s2m),sE2),sE1)
            r[0,1] = 0.0
            r[1,1] = 0.0
            r[2,1] = 1.0

        hbar =  max(0.5 * (hLstar + hRstar), 0.0)
        s1s2bar = 0.25 * (uLstar + uRstar)**2 - g * hbar
        s1s2tilde = max(0.0, uLstar * uRstar) - g * hbar

        # find if sonic problem
        sonic = False
        if (np.abs(s1s2bar) <= criticaltol): sonic = True
        if (s1s2bar * s1s2tilde <= criticaltol**2): sonic = True
        if (s1s2bar * sE1 * sE2 <= criticaltol**2): sonic = True
        if (min(np.abs(sE1), np.abs(sE2)) < criticaltol_2): sonic = True
        if (sE1 < criticaltol_2 and s1m > -criticaltol_2): sonic = True
        if (sE2 > -criticaltol_2 and s2m < criticaltol_2): sonic = True
        if ((uL + np.sqrt(g * hL)) * (uR + np.sqrt(g * hR)) < 0): sonic = True
        if ((uL - np.sqrt(g * hL)) * (uR - np.sqrt(g * hR)) < 0): sonic = True

        # find jump in h, deldelh
        if (sonic):
            deldelh = -delb
        else:
            deldelh = delb * g * hbar / s1s2bar

        # find bounds in case of critical state resonance, or negative states
        if (sE1 < -criticaltol and sE2 > criticaltol):
            deldelh = min(deldelh, hstarHLL * (sE2 - sE1) / sE2)
            deldelh = max(deldelh, hstarHLL * (sE2 - sE1) / sE1)
        elif (sE1 >= criticaltol):
            deldelh = min(deldelh, hstarHLL * (sE2 - sE1) / sE1)
            deldelh = max(deldelh, -hL)
        elif (sE2 <= -criticaltol):
            deldelh = min(deldelh, hR)
            deldelh = max(deldelh, hstarHLL * (sE2 - sE1) / sE2)

        # find jump in phi, deldelphi
        if (sonic):
            deldelphi = -g * hbar * delb
        else:
            deldelphi = -delb * g * hbar * s1s2tilde / s1s2bar

        # find bounds in case of critical state resonance, or negative states
        deldelphi = min(deldelphi, g * max(-hLstar * delb, -hRstar * delb))
        deldelphi = max(deldelphi, g * min(-hLstar * delb, -hRstar * delb))

        dell[0] = delh - deldelh
        dell[1] = delhu
        dell[2] = delphi - deldelphi

        # Determine determinant of eigenvector matrix
        det1 = r[0,0] * (r[1,1] * r[2,2] - r[1,2] * r[2,1])
        det2 = r[0,1] * (r[1,0] * r[2,2] - r[1,2] * r[2,0])
        det3 = r[0,2] * (r[1,0] * r[2,1] - r[1,1] * r[2,0])
        determinant = det1-det2+det3

        # solve for beta(k) using Cramers Rule=================
        for k in range (num_waves):
            for mw in range (num_waves):
                A[0,mw] = r[0, mw]
                A[1,mw] = r[1, mw]
                A[2,mw] = r[2, mw]
            A[0,k] = dell[0]
            A[1,k] = dell[1]
            A[2,k] = dell[2]
            det1 = A[0,0] * (A[1,1] * A[2,2] - A[1,2] * A[2,1])
            det2 = A[0,1] * (A[1,0] * A[2,2] - A[1,2] * A[2,0])
            det3 = A[0,2] * (A[1,0] * A[2,1] - A[1,1] * A[2,0])
            beta[k] = (det1-det2+det3)/determinant

        # exit if things aren't changing

        if (np.abs(dell[0]**2 + dell[2]**2 - delnorm) < convergencetol): break
        delnorm = dell[0]**2 + dell[2]**2
        # find new states qLstar and qRstar on either side of interface
        hLstar = hL
        hRstar = hR
        uLstar = uL
        uRstar = uR
        huLstar = uLstar * hLstar
        huRstar = uRstar * hRstar
        for mw in range (num_waves):
            if (lamda[mw] < 0.0):
                hLstar = hLstar + beta[mw] *r[0,mw]
                huLstar = huLstar + beta[mw] * r[1,mw]
        for mw in range (num_waves-1,-1,-1):
            if (lamda[mw] > 0.0):
                hRstar = hRstar - beta[mw] * r[0,mw]
                huRstar = huRstar - beta[mw] * r[1,mw]

        if (hLstar > drytol):
            uLstar = huLstar / hLstar
        else:
            hLstar = max(hLstar, 0.0)
            uLstar = 0.0

        if (hRstar > drytol):
            uRstar = huRstar / hRstar
        else:
            hRstar = max(hRstar, 0.0)
            uRstar = 0.0

    # end iteration on Riemann problem

    for mw in range (num_waves):
        sw[mw] = lamda[mw]
        fw[0,mw] = beta[mw] * r[1,mw]
        fw[1,mw] = beta[mw] * r[2,mw]

    return sw, fw



def shallow_JCP_dry_1d(q_l, q_r, aux_l, aux_r, problem_data):

    g = problem_data['grav']
    # nw = problem_data['wall_position']
    # wall_height = problem_data['wall_height']
    drytol = problem_data['dry_tolerance']
    maxiter = problem_data['max_iteration']
    # alpha = problem_data['fraction']

    num_rp = q_l.shape[1]
    num_eqn = 2
    num_waves = 3

    # Output arrays
    fwave = np.zeros( (num_eqn, num_waves, num_rp) )
    s = np.zeros( (num_waves, num_rp) )
    amdq = np.zeros( (num_eqn, num_rp) )
    apdq = np.zeros( (num_eqn, num_rp) )
    # sw = np.zeros(num_waves +1)
    # fw = np.zeros(num_eqn, num_waves+1) )

    for i in range(num_rp):
        hL = q_l[0,i]
        hR = q_r[0,i]
        huL = q_l[1,i]
        huR = q_r[1,i]
        bL = aux_l[0,i]
        bR = aux_r[0,i]

        # Check wet/dry states
        if (hR > drytol): # right state is not dry
            uR = huR / hR
            phiR = 0.5 * g * hR**2 + huR**2 / hR
        else:
            hR = 0.0
            huR = 0.0
            uR = 0.0
            phiR = 0.0

        if (hL > drytol):
            uL = huL / hL
            phiL = 0.5 * g * hL**2 + huL**2 / hL
        else:
            hL = 0.0
            huL = 0.0
            uL = 0.0
            phiL = 0.0

        if (hL > drytol or hR > drytol):
            wall = np.ones(3)
            if (hR <= drytol):
                hstar,_,_,_,_= riemanntype(hL, hL, uL, -uL, maxiter, drytol, g)
                hstartest = max(hL, hstar)
                # hstartest = hstar
                if (hstartest + bL < bR):
                    wall[1] = 0.0
                    wall[2] = 0.0
                    hR = hL
                    huR = -huL
                    bR = bL
                    phiR = phiL
                    uR = -uL
                elif (hL + bL < bR):
                    bR = hL + bL

            elif (hL <= drytol):
                hstar,_,_,_,_= riemanntype(hR, hR, -uR, uR, maxiter, drytol, g)
                hstartest = max(hR, hstar)
                # hstartest = hstar
                if (hstartest + bR < bL):
                    print("the wall is high enough")
                    wall[1] = 0.0
                    wall[2] = 0.0
                    hL = hR
                    huL = -huR
                    bL = bR
                    phiL = phiR
                    uL = -uR
                elif (hR+ bR < bL):
                    bL = hR + bR

            sL = uL - np.sqrt(g * hL)
            sR = uR + np.sqrt(g * hR)
            uhat = (np.sqrt(g * hL) * uL + np.sqrt(g * hR) * uR) / (np.sqrt(g * hR) + np.sqrt(g * hL))
            chat = np.sqrt(g * 0.5 * (hR + hL))
            sRoe1 = uhat - chat
            sRoe2 = uhat + chat
            sE1 = min(sL, sRoe1)
            sE2 = max(sR, sRoe2)
            sw, fw = riemann_aug_JCP(maxiter,hL,hR,huL,huR,bL,bR,uL,uR,phiL,phiR,sE1,sE2,drytol,g)

            for mw in range(num_waves):
                s[mw,i] = sw[mw] * wall[mw]
                fwave[:,mw,i] = fw[:,mw] * wall[mw]

            for mw in range(num_waves):
                if (s[mw,i] < 0):
                    amdq[:,i] += fwave[:,mw,i]
                elif (s[mw,i] > 0):
                    apdq[:,i] += fwave[:,mw,i]
                else:
                    amdq[:,i] += 0.5 * fwave[:,mw,i]
                    apdq[:,i] += 0.5 * fwave[:,mw,i]

    return fwave, s, amdq, apdq
