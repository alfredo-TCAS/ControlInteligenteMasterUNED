import numpy as np
import scipy as sp

from uav_model import uav_model
from uav_model import param_evolve
from guidance import serret_frenet_angle
from guidance import LOS_angle
from guidance import uav_SF_angle

# Just a simple Euler integrator
def simulator_v2(x0,p0,s,Ts,tmax,tsplines,xsplines,ysplines,kp,ki,kd,ks):
    niter = int(tmax // Ts)
    
    # Curves
    curve = curves()

    # init pid
    pid = pid_class(kp=kp,ki=ki,kd=kd,max_sum=400,dt=Ts)
    
    
    # For later info
    uav_inf = uav_info(s)
    pp = []; tt = []
    
    for k in range(niter):
    
        # do it better
        if(p0 > tsplines[-1]): 
            break
            
        px = x0[0]; py = x0[1];
        psi = x0[2]; omega = x0[3]
        curve.eval_spline(tsplines, 
                          xsplines,ysplines,
                          p0)
        
        f = curve.f; df = curve.df;
    
        eN = f[0] - px # x
        eT = f[1] - py # y
        psiLOS = LOS_angle(eN,eT)
        psiT   = serret_frenet_angle(df[0],df[1])
        psiSF  = uav_SF_angle(psi,psiT)
    
        """ If we follow LOS, we converge to SF"""
        e = (psiLOS - psi)
        u = pid.compute_pid(e)
        # saturate u
        if(u >= 2*np.pi):
            u = 2*np.pi
        elif(u <= -2*np.pi):
            u = -2*np.pi;
            
        ed = np.sqrt( eN**2 + eT**2)
        
        x = np.array([px,py,psi,omega])
        
        xdot_uav = uav_model(k*Ts, x, s, u)
        #xdot_frenet = param_evolve(k*Ts, 0, s, psiSF, ks, eN**2, df)
        xdot_frenet = param_evolve(k*Ts, 0, s, psiSF, ks, eT, df)

        # integrate by Euler
        x0 = x0 + xdot_uav * Ts;
        p0 = p0 + xdot_frenet * Ts
        
        #res = sp.integrate.solve_ivp(uav_model, [k*Ts,(k+1)*Ts], 
        #                             x0, args=(s,u),
        #                             dense_output=True,
        #                             max_step=0.1*Ts)
        #resp = sp.integrate.solve_ivp(param_evolve, [k*Ts, (k+1)*Ts],
        #                              p0, args=(s,psiSF,ks,eT,df),
        #                              dense_output=True,
        
        #                              max_step=0.1*Ts)
        # save states and error
        uav_inf.append(px,py,psi,omega,u,e,ed,psiSF)
        pp = np.append(pp,p0)
        tt = np.append(tt,k*Ts)
        #x0 = res.y[:,-1]
        #p0 = resp.y[:,-1]
    
    return tt, uav_inf, pp   

# Runge kutta integrator
def simulator(x0,p0,s,Ts,tmax,tsplines,xsplines,ysplines,kp,ki,kd,ks):

    niter = int(tmax // Ts)
    
    # Curves
    curve = curves()

    # init pid
    pid = pid_class(kp=kp,ki=ki,kd=kd,max_sum=100,dt=Ts)
    
    # For later info
    uav_inf = uav_info(s)
    pp = []; tt = []
    
    for k in range(niter):
    
        # do it better
        if(p0 > tsplines[-1]): 
            break
            
        px = x0[0]; py = x0[1];
        psi = x0[2]; omega = x0[3]
        curve.eval_spline(tsplines, 
                          xsplines,ysplines,
                          p0)
        
        f = curve.f; df = curve.df;
    
        eN = f[0] - px # x
        eT = f[1] - py # y
        psiLOS = LOS_angle(eN,eT)
        psiT   = serret_frenet_angle(df[0],df[1])
        psiSF  = uav_SF_angle(psi,psiT)
    
        """ If we follow LOS, we converge to SF"""
        e = (psiLOS - psi)
        u = pid.compute_pid(e)
        
        res = sp.integrate.solve_ivp(uav_model, [k*Ts,(k+1)*Ts], 
                                     x0, args=(s,u),
                                     dense_output=True,
                                     max_step=0.1*Ts)
        resp = sp.integrate.solve_ivp(param_evolve, [k*Ts, (k+1)*Ts],
                                      p0, args=(s,psiSF,ks,eT,df),
                                      dense_output=True,
                                      max_step=0.1*Ts)
        # save states and error
        uav_inf.append(px,py,psi,omega,u,e)
        pp = np.append(pp,p0)
        tt = np.append(tt,k*Ts)
        x0 = res.y[:,-1]
        p0 = resp.y[:,-1]
    return tt, uav_inf, pp
    
class curves:
    def __init__(self):
        self.supported_curves = ['circle']
        self.f = np.array([0,0])
        self.df = np.array([0,0])
    def eval_circle(self,radius,center,angle):
        x = radius * np.cos(angle) + center[0]
        y = radius * np.sin(angle) + center[1]
        dx = -radius * np.sin(angle)
        dy = radius * np.cos(angle)
        self.f[0] = x; self.f[1] = y;
        self.df[0] = dx; self.df[1] = dy;

    def eval_spline(self, t, px, py, w):
        cur = sp.interpolate.CubicSpline(t, np.c_[px,py],bc_type="natural")
        self.f[0],self.f[1] = cur(w).T
        self.df[0], self.df[1] = cur.derivative(1)(w).T



"""
def eval_circle(radius,center,angle):
    x = radius * np.cos(angle) + center[0]
    y = radius * np.sin(angle) + center[1]
    dx = -radius * np.sin(angle)
    dy = radius * np.cos(angle)
    # return circle and its derivative
    return np.array([x,y]), np.array([dx,dy])
"""
class uav_info:
    def __init__(self,airspeed=1):
        self.px = []
        self.py = []
        self.psi = []
        self.omega = []
        self.s = airspeed
        self.u = [] # control action
        self.e = [] # orientation error w.r.t LOS
        self.ed = [] # distance error
        self.eSF = [] # oerinetation error w.r.t serret frenet
    def append(self,px,py,psi,omega,u,e,ed,eSF):
        self.px = np.append(self.px,px)
        self.py = np.append(self.py,py)
        self.psi = np.append(self.psi,psi)
        self.omega = np.append(self.omega,omega)
        self.u = np.append(self.u,u)
        self.e = np.append(self.e,e)    
        self.ed = np.append(self.ed,ed)
        self.eSF = np.append(self.eSF,eSF)
        
class pid_class:
    def __init__(self,kp,ki,kd,max_sum,dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.sum = 0
        self.dt = dt
        # Maximal sum for the integrator (saturate, avoid windup)
        self.max_sum = max_sum
        self.e = np.array([0,0])
    def update_pid(self,kp,ki,kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
    def compute_pid(self,err):
        # update error
        self.e[1] = self.e[0]
        self.e[0] = err
        # proportional
        up = self.kp * err
        # derivative
        ud = self.kd * (self.e[0] - self.e[1]) / self.dt
        # integral 
        integral = self.ki * (self.sum + self.e[0] * self.dt)
        #self.sum = self.sum + self.e[0] * self.dt
        if(integral > self.max_sum):
            integral = self.max_sum
        elif(integral < -self.max_sum):
            integral = -self.max_sum
        else:
            self.sum += self.e[0] * self.dt
       # else:
        #    self.sum = self.sum + self.e[0] * self.dt
        # return PID
        # THIS IS NOT OK. CHECK THIS TOMORROW
        return up + integral + ud
            
        