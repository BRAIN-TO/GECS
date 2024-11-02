from models.pdcm import pdcm_fmri_priors_new
from models.hemodynamic import *
from io import loadmat
import numpy as np
from spm.nlsi import spm_nlsi_GN
from spm.integrate import spm_int_IT
import matplotlib.pyplot as plt
from plot_utils import PlotGraph
import warnings
warnings.filterwarnings('ignore')


SPM = loadmat('SPM.mat')

Y = {'y':np.zeros((360,3))}
V1 = loadmat('VOI_V1_1.mat')
Y['y'][:,0] = V1['xY']['u'][:,0]
V5 = loadmat('VOI_V5_1.mat')
Y['y'][:,1] = V5['xY']['u'][:,0]
SPC = loadmat('VOI_SPC_1.mat')
Y['y'][:,2] = SPC['xY']['u'][:,0]
Y['dt'] = SPM['SPM']['xY']['RT']
Y['X0'] = np.concatenate((np.ones((V1['xY']['X0'].shape[0],1)),V1['xY']['X0'][:,1:6]), axis=1)

scale   = np.max(Y['y']) - np.min(Y['y'])
scale   = 4/max(scale,4)
Y['y']     = Y['y']*scale
Y['scale'] = scale

u_idx = [1, 2, 0]
Sess   = SPM['SPM']['Sess']
U = {}
U['name'] = []
U['u'] = np.zeros((5760, 1))
for i in range(0,len(u_idx)):
    u = u_idx[i]
    for j in range(0,1):
        U['u']             = np.concatenate((U['u'], np.expand_dims(Sess['U'][u]['u'][32:,j], 1)), axis=1)
        U['name'].append(Sess['U'][u]['name'][0])

U['u'] = U['u'][:,1:]        
U['dt']   = Sess['U'][0]['dt']
U['u'][U['u']>1] = 1

A = np.array([[0, 1, 0],
          [1, 0, 1],
          [0, 1, 0]])
B = np.array([[[0, 0, 0],
        [1, 0, 0],
        [0, 0, 0]],

       [[0, 0, 0],
        [0, 0, 1],
        [0, 0, 0]]])
    
C = np.array([[0, 0, 1],
          [0, 0, 0],
          [0, 0, 0]])

D = np.array([[0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]])
    
pE, pC, x, _ = pdcm_fmri_priors_new(A, B, C, D, {'decay':1})

M = {}
B0      = 3
TE      = 0.04
nr      = Y['y'].shape[1]
M['delays'] = np.ones((1,nr))*Y['dt']/2 
M['TE']    = TE
M['B0']    = B0
M['m']     = nr
M['n']     = 6         
M['l']     = nr
M['N']     = 64
M['dt']    = U['dt']
M['ns']    = Y['y'].shape[0]
M['x']     = x
M['IS']    = 'spm.integ_IT'

M['f']   = 'fx_fmri_pdcm_new'
M['g']   = 'gx_all_fmri'
M['Tn']  = []                     
M['Tc']  = []
M['Tv']  = []
M['Tm']  = []

n           = nr

M['pE'] = pE
M['pC'] = pC

Ep,Cp,Eh,F,L,dFdp,dFdpp = spm_nlsi_GN(M, U, Y)

y = spm_int_IT(Ep, M, U)