import numpy as np
from scipy import interpolate
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

import quaternion_dmp

class CartesianDMP():
    
    def __init__(self,N_bf=20,alphaz=4.0,betaz=1.0):
        
        self.alphax = 1.0
        self.alphaz = alphaz
        self.betaz = betaz
        self.N_bf = N_bf # number of basis functions
        self.tau = 1.0 # temporal scaling

        # Orientation dmp
        self.dmp_ori = quaternion_dmp.QuaternionDMP(self.N_bf,self.alphax,self.alphaz,self.betaz,self.tau)

    def imitate(self, pose_demo, sampling_rate=100, oversampling=True):
        
        self.T = pose_demo.shape[0] / sampling_rate
        
        if not oversampling:
            self.N = pose_demo.shape[0]
            self.dt = self.T / self.N
            self.x = pose_demo[:,:3]
            
        else:
            self.N = 10 * pose_demo.shape[0] # 10-fold oversample
            self.dt = self.T / self.N

            t = np.linspace(0.0,self.T,pose_demo[:,0].shape[0])
            self.x = np.zeros([self.N,3])
            for d in range(3):
                x_interp = interpolate.interp1d(t,pose_demo[:,d])
                for n in range(self.N):
                    self.x[n,d] = x_interp(n * self.dt)
                
        # Centers of basis functions 
        self.c = np.ones(self.N_bf) 
        c_ = np.linspace(0,self.T,self.N_bf)
        for i in range(self.N_bf):
            self.c[i] = np.exp(-self.alphax *c_[i])

        # Widths of basis functions 
        # (as in https://github.com/studywolf/pydmps/blob/80b0a4518edf756773582cc5c40fdeee7e332169/pydmps/dmp_discrete.py#L37)
        self.h = np.ones(self.N_bf) * self.N_bf**1.5 / self.c / self.alphax

        self.dx = np.gradient(self.x,axis=0)/self.dt
        self.ddx = np.gradient(self.dx,axis=0)/self.dt

        # Initial and final orientation
        self.x0 = self.x[0,:]
        self.dx0 = self.dx[0,:] 
        self.ddx0 = self.ddx[0,:]
        self.xT = self.x[-1,:]

        # Evaluate the phase variable
        self.phase = np.exp(-self.alphax*np.linspace(0.0,self.T,self.N))

        # Evaluate the forcing term
        forcing_target_pos = self.tau*self.ddx - self.alphaz*(self.betaz*(self.xT-self.x) - self.dx)

        self.fit_dmp(forcing_target_pos)
        
        # Imitate orientation
        q_des = self.dmp_ori.imitate(pose_demo[:,3:], sampling_rate, oversampling)
        
        return self.x, q_des
    
    def RBF(self):
        return np.exp(-self.h*(self.phase[:,np.newaxis]-self.c)**2)

    def forcing_function_approx(self,weights,phase,xT=1,x0=0):
        BF = self.RBF()
        return np.dot(BF,weights)*phase*(xT-x0)/np.sum(BF,axis=1)
    
    def fit_dmp(self,forcing_target):

        BF = self.RBF()
        X = BF*self.phase[:,np.newaxis]/np.sum(BF,axis=1)[:,np.newaxis]

        self.weights_pos = np.zeros([self.N_bf,3])
        for d in range(3):
            self.weights_pos[:,d] = np.dot(np.linalg.pinv(X*(self.xT[d]-self.x0[d])),forcing_target[:,d])

    def rollout(self,tau=1.0,xT=None):

        x_rollout = np.zeros([self.N,3])
        dx_rollout = np.zeros([self.N,3])
        ddx_rollout = np.zeros([self.N,3])
        x_rollout[0,:] = self.x0
        dx_rollout[0,:] = self.dx0
        ddx_rollout[0,:] = self.ddx0
        
        if xT is None:
            xT = self.xT
        
        phase = np.exp(-self.alphax*tau*np.linspace(0.0,self.T,self.N))

        # Position forcing term
        forcing_term_pos = np.zeros([self.N,3])
        for d in range(3):
            forcing_term_pos[:,d] = self.forcing_function_approx(
                self.weights_pos[:,d],phase,xT[d],self.x0[d])

        for d in range(3):
            for n in range(1,self.N):
                ddx_rollout[n,d] = self.alphaz*(self.betaz*(xT[d]-x_rollout[n-1,d]) - \
                                               dx_rollout[n-1,d]) + forcing_term_pos[n,d]
                dx_rollout[n,d] = dx_rollout[n-1,d] + tau*ddx_rollout[n-1,d]*self.dt
                x_rollout[n,d] = x_rollout[n-1,d] + tau*dx_rollout[n-1,d]*self.dt
        
        # Get orientation rollout
        q_rollout,dq_log_rollout,ddq_log_rollout = self.dmp_ori.rollout(tau=tau)
        
        return x_rollout,dx_rollout,ddx_rollout, q_rollout,dq_log_rollout,ddq_log_rollout


# Test

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    with open('pose_trajectory.npy', 'rb') as f:
        demo_trajectory = np.load(f)

    position_trajectory = np.zeros([len(demo_trajectory),3])
    orientation_trajectory = np.zeros([len(demo_trajectory),4])
    for i in range(demo_trajectory.shape[0]):
        position_trajectory[i,:] = demo_trajectory[i][:3,3]
        orientation_trajectory[i,:] = R.from_matrix(demo_trajectory[i][:3,:3]).as_quat()

    pose_trajectory = np.hstack((position_trajectory,orientation_trajectory))

    dmp = CartesianDMP()
    x_des, q_des = dmp.imitate(pose_trajectory)

    goal_offset = [0.01,0.01,0.0] # new position goal
    x_rollout, dx_rollout, _, q_rollout, dq_rollout, _ = dmp.rollout(xT=dmp.xT+goal_offset)

    fig = plt.figure(figsize=(21,3))
    for d in range(3):
        plt.subplot(131+d)
        plt.plot(x_des[:,d],label='demo')
        plt.plot(x_rollout[:,d],'--',label='rollout')
    plt.suptitle('Position trajectory')
    plt.show()

    fig = plt.figure(figsize=(35,3))
    for d in range(4):
        plt.subplot(141+d)
        plt.plot(q_des[:,d],label='demo')
        plt.plot(q_rollout[:,d],'--',label='rollout')
    plt.suptitle('Orientation trajectory')
    plt.show()