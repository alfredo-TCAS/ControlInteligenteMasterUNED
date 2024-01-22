import numpy as np

"""
    Param:
        @ fx : curve evaluated at a point w
        @ fy : curve evaluated at a point w
        @ w  : parameter of the curve
"""
def serret_frenet_angle(dfx,dfy):
    # If not normalized is ok, because the division of arctangent removes that
    return np.arctan2(dfy,dfx)

def LOS_angle(los_x,los_y):
    return np.arctan2(los_y,los_x)

def uav_SF_angle(psi,psiT):
    return psiT - psi
    