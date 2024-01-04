import numpy as np

""" Function uav_model
    @ t : time
    @ x : states (not z implemented)
    @ s : airspeed
    @ u : control actiosn
    @ w : wind
"""
def uav_model(t,x,s,u,w=[0,0]):

    # Control function: angular rate
    upsi = u
    # states
    px = x[0]
    py = x[1]
    psi = x[2]   
    omegapsi = x[3]
    # derivative
    xdot = np.zeros(x.shape)
    xdot[0] = s * np.cos(psi) + w[0]
    xdot[1] = s * np.sin(psi) + w[1]
    xdot[2] = omegapsi
    xdot[3] = upsi - omegapsi*10 # rozamiento en la señal de control (rozamiento lateral)
    #xdot[3] = uz
    return xdot

def param_evolve(t,x,s,psiSF,ks,eT,df):

    dp = (s*np.cos(psiSF) + ks*eT)/np.sqrt(df[0]**2+df[1]**2)
    return dp
    