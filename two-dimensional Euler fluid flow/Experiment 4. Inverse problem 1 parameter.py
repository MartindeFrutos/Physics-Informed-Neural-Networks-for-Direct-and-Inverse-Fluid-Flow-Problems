# Experiment 4. Inverse problem 1 parameter 
import os 
import torch 
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import pandas as pd 
import pickle 
from Domain_classes import Domain, Distance , Boundary 
from PINN_forward_classes import PINN_2DFlow
from PINN_inverse_classes import PINN2D_Inverse_1parameter

mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size': 13
})

# Directories
folder='4. Inverse problem 1 parameter data/'
tex_folder=folder+'PGF/'
distance_name=folder+'Trained_distance. 1 parameter.pt'
PINN_name=folder+'PINN inverse 1 parameter ANIMATION.pt'
# PINN_name=folder+'PINN inverse 1 parameter - AL=2.78.pt'
forward_path='C:/Users/marti/Desktop/MUMAC/TFM/2 Dim Flow/Comparison 2D vs 1D/'

# Do we want to retrain for some reason ? 
RETRAIN=True

RETRAIN_DISTANCE=False

# Initial Geometry (AL=1.5)
Ix=[0.0,4.0]
f1 = lambda x : 0.5+0.25/(1+torch.exp(-2.0*(x-Ix[1]*0.5-Ix[0]*0.5)))
f2 = lambda x : -f1(x)

Dom=Domain(Ix,f1,f2)

# Computing the initial distance function 
Dist=Distance(Ix,f1,f2)
if RETRAIN_DISTANCE==False and os.path.exists(distance_name):
    Dist.load_state_dict(torch.load(distance_name))
else:
    Dist.fit()
    torch.save(Dist.state_dict(),distance_name)
    
Dom.figure(Dist.forward,r'$D_\varphi(\bar x)$','Pretrained distance.pgf',tex_folder)
plt.scatter(Dist.X3[:,0],Dist.X3[:,1],marker='x',color='red',label=r'Training points')
plt.legend()
plt.savefig(folder+'Pretrained distance 2.svg', format='svg')
plt.close()
if os.path.exists(tex_folder+'CMAP.png'):
    os.remove(tex_folder+'Pretrained distance-img1.png')
else : 
    os.rename(tex_folder+'Pretrained distance-img1.png',tex_folder+'CMAP.png')

# Forward solution 
AL_SOL=3.0
f1sol = lambda x : 0.5+(3/2-0.5)/(1+torch.exp(-2.0*(x-2)))
f2sol = lambda x : -f1sol(x) 
D_Forward=Distance(Ix,f1sol,f2sol)
D_Forward.load_state_dict(torch.load(forward_path+'Trained_distance.pt'))
G_Forward=Boundary(0.0,Ix,f1sol,f2sol)
G_Forward.load_state_dict(torch.load(forward_path+'Trained_boundary.pt'))
N2=PINN_2DFlow([2,50,20,50,4],D_Forward,G_Forward)
N2.load_state_dict(torch.load(forward_path+'2DPINN.pt'))
N2.eval()
X , rho , u , v , p , vmod  , theta = N2.Solution(N=10000) 
x=torch.linspace(N2.Distance.Ix[0],N2.Distance.Ix[1],100).reshape(100,1)
M_forward=torch.zeros_like(x)
u_forward=torch.zeros_like(x)
rho_forward=torch.zeros_like(x)
p_forward=torch.zeros_like(x)
for i , xc in enumerate(x):
    rho_forward[i],u_forward[i],p_forward[i],M_forward[i]=N2.One_Dimensional_results(xc)
    
# Inverse problem solver  
NL=[2,50,20,50,4]
N=PINN2D_Inverse_1parameter(NL,Dom,Dist,folder,animation=True)
if RETRAIN==False and os.path.exists(PINN_name):# Loading
    N.load_state_dict(torch.load(PINN_name))
else: # Training 
    # We remove the old summarywriter files
    for file in os.listdir(folder+'logs'):
        os.remove(folder+'logs/'+file)
    # Training of the PINN inverse model 
    N.train()
    N.fit()
    # Save the results 
    torch.save(N.state_dict(),PINN_name)
    if N.animation==True:
        with open(folder+'VARIABLES.pickle', "wb") as archivo:
            pickle.dump(N.VARIABLES, archivo)

print('The value of the parameter is : ', float(N.AL.data[0]))
N.Domain.f1=lambda x : N.f1(x,N.AL)
N.Domain.f2=lambda x : N.f2(x,N.AL)

# Results
N.eval()

# Some interior points results 
X=N.Domain.Construct_points(10000)
X , rho , u , v , p , vmod  , theta = N.Solution(X) 

# ONE DIMENSIONAL RESULTS 
x=torch.linspace(N.Domain.Ix[0],N.Domain.Ix[1],100).reshape(100,1)
M=torch.zeros_like(x)
u_mean=torch.zeros_like(x)
rho_mean=torch.zeros_like(x)
p_mean=torch.zeros_like(x)
for i , xc in enumerate(x):
    rho_mean[i],u_mean[i],p_mean[i],M[i]=N.One_Dimensional_results(xc)

# Mass flow 
plt.figure()
plt.xlabel(r'$x$')
plt.ylabel(r'$\dot{m}$')
plt.ylim([0.5,1.5])
with torch.no_grad():
    plt.plot(x,M,label='Inverse model')
    plt.plot(x,M_forward,'--',label='Forward reference')
    plt.plot(x,0*x+N.Domain.f1(0)*2,linestyle='dashdot',label='Constant')
plt.legend()
plt.savefig(folder+'Mass flow.svg', format='svg')
plt.savefig(tex_folder+'Mass flow check.pgf',dpi=1000)
plt.close() 

# Horizontal velocity 
fig=plt.figure()
plt.xlabel(r'$x$')
plt.ylabel(r'$\tilde{u}$')
with torch.no_grad():
    plt.plot(x,u_mean,label='Inverse model')
    plt.plot(x,u_forward,'--',label='Forward reference')
plt.legend(loc='lower left')
fig.savefig(folder+'Horizontal velocity comparison.svg', format='svg')
fig.savefig(tex_folder+'Horizontal velocity comparison.pgf',dpi=1000)
plt.close()

# Average pressure 
fig=plt.figure()
plt.xlabel(r'$x$')
plt.ylabel(r'$\tilde{p}$')
with torch.no_grad():
    plt.plot(x,p_mean,label='Inverse model')
    plt.plot(x,p_forward,'--',label='Forward reference')
plt.legend()
fig.savefig(folder+'Pressure comparison.svg', format='svg')
fig.savefig(tex_folder+'Pressure comparison.pgf', dpi=1000)
plt.close()

# Average density 
fig=plt.figure()
plt.xlabel(r'$x$')
plt.ylabel(r'$\tilde\rho$')
with torch.no_grad():
    plt.plot(x,rho_mean,label='Inverse model')
    plt.plot(x,rho_forward,'--',label='Forward reference')
plt.legend()
fig.savefig(folder+'Average Density comparison.svg', format='svg')
fig.savefig(tex_folder+'Average Density comparison.pgf',dpi=1000)

# TWO DIMENSIONAL RESULTS 

# Distance on final geometry
N.Domain.figure(lambda X : N.Smooth_distance(X)[2],r'$D_\varphi(\bar x)$','Final distance.pgf',tex_folder)
plt.legend()
plt.savefig(folder+'Final distance.svg', format='svg')
plt.savefig(tex_folder+'Final distance.pgf', dpi=1000)
plt.close()
os.remove(tex_folder+'Final distance-img1.png')

# DENSITY
N.Domain.figure(lambda X : N.Solution(X)[1],r'$\rho(\bar x)$','Density 2D.pgf',tex_folder)
plt.savefig(folder+'Density 2D.svg', format='svg')
os.remove(tex_folder+'Density 2D-img1.png')
plt.close()

# PRESSURE    
N.Domain.figure(lambda X : N.Solution(X)[4],r'$p(\bar x)$','Pressure 2D.pgf',tex_folder)
plt.savefig(folder+'Pressure 2D.svg', format='svg')
os.remove(tex_folder+'Pressure 2D-img1.png')
plt.close()


# VELOCITY 
X , rho , u , v , p , vmod , theta = N.Solution(N=300) # Some interior points results 
fig=plt.figure()
plt.title(r'Velocity field')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
with torch.no_grad():
    norm= mcolors.Normalize(vmin=0,vmax=torch.max(vmod))
    plt.quiver(X[:,0],X[:,1],u,v,vmod,cmap='jet',norm=norm,
                scale_units='xy',scale=0.75, width=0.003, 
                headwidth=4)
    x=torch.linspace(N.Domain.Ix[0],N.Domain.Ix[1],100)
    plt.plot(x,N.Domain.f1(x),color='black',label=r'$\partial\Omega$')
    plt.plot(x,N.Domain.f2(x),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$||\bar v(\bar x)||_2$')
plt.legend()
plt.tight_layout()
fig.savefig(folder+'Velocity field.svg', format='svg') 
plt.savefig(tex_folder+'Velocity field.pgf',dpi=1000)
plt.close()

            
#  Convergence plots 
dataAL=pd.read_csv(folder+'AL.csv')
dataL=pd.read_csv(folder+'L.csv')
dataL1=pd.read_csv(folder+'L1.csv')
dataL2=pd.read_csv(folder+'L2.csv')
dataL3=pd.read_csv(folder+'L3.csv')
dataL4=pd.read_csv(folder+'L4.csv')
dataLU=pd.read_csv(folder+'L5.csv')

plt.figure()
plt.xlabel('Iterations')
plt.ylabel('Loss function')
plt.semilogy(dataL['Step'],dataL['Value'],linewidth=0.75,label=r'$\mathcal L$')
plt.semilogy(dataL['Step'],dataL1['Value'],linewidth=0.75,label=r'$\mathcal L_1$')
plt.semilogy(dataL['Step'],dataL2['Value'],linewidth=0.75,label=r'$\mathcal L_2$')
plt.semilogy(dataL['Step'],dataL3['Value'],linewidth=0.75,label=r'$\mathcal L_3$')
plt.semilogy(dataL['Step'],dataL4['Value'],linewidth=0.75,label=r'$\mathcal L_4$')
plt.semilogy(dataL['Step'],dataLU['Value'],linewidth=0.75,label=r'$\mathcal L_U$')
plt.legend()
plt.tight_layout()
plt.savefig(folder+'Losses.svg',format='svg')
plt.savefig(tex_folder+'Losses.pgf',dpi=1000)
plt.close()

plt.figure()
plt.xlabel('Iterations')
plt.ylabel(r'$A_L$')
plt.plot(dataAL['Step'],dataAL['Value'])
plt.savefig(folder+'Parameter evolution.svg',format='svg')
plt.savefig(tex_folder+'Parameter evolution.pgf',dpi=1000)
# plt.close()

