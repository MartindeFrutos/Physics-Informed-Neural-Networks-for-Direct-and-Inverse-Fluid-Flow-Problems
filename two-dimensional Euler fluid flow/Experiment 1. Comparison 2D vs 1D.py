# Experiment 1: Comparing the two-dimensional solver with the one-dimensional solver. 
import os 
import torch 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
from Domain_classes import Domain, Distance , Boundary 
from PINN_forward_classes import PINN_2DFlow , PINN_1DFlowA

mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size': 13
})

# Directories
folder='1. Comparison 2D vs 1D/'
folder_svg=folder+'SVG figures/'
folder_pgf=folder+'PGF figures/'
distance_name=folder+'Trained_distance.pt'
boundary_name=folder+'Trained_boundary.pt'
PINN1_name=folder+'1DPINN.pt'
PINN2_name=folder+'2DPINN.pt'

# Do we want to retrain for some reason ? 
RETRAIN=False

# Geometry 
Ix=[0.0,4.0]
f1 = lambda x : 0.5+1.0/(1+torch.exp(-2*(x-Ix[1]*0.5-Ix[0]*0.5)))
f2 = lambda x : -f1(x)

# Domain pretraining
Dom=Domain(Ix,f1,f2)

# Computing the distance function 
Dist=Distance(Ix,f1,f2)
if RETRAIN==False and os.path.exists(distance_name):
    Dist.load_state_dict(torch.load(distance_name))
else:
    Dist.fit()
    torch.save(Dist.state_dict(),distance_name)

# Computing the boundary function
G=Boundary(0.0,Ix,f1,f2)
if RETRAIN==False and os.path.exists(boundary_name):
    G.load_state_dict(torch.load(boundary_name))
else:
    G.fit()
    torch.save(G.state_dict(),boundary_name)

# Plots 
X=Dom.Construct_points(10000)

# Distance plot 
Dom.figure(Dist.forward,r'$D_\varphi(\bar x)$','Trained distance.pgf',folder_pgf)
plt.close()
if os.path.exists(folder_pgf+'CMAP.png'):
    os.remove(folder_pgf+'Trained distance-img1.png')
else : 
    os.rename(folder_pgf+'Trained distance-img1.png',folder_pgf+'CMAP.png')

# Boundary plot 
Dom.figure(G.forward,r'$G_\varphi(\bar x)$','Boundary extension.pgf',folder_pgf)
os.remove(folder_pgf+'Boundary extension-img1.png')
plt.close()

# 1 Dimension Simulation 
NL1=[1,50,20,50,3]
N1=PINN_1DFlowA(NL1)
if RETRAIN==False and os.path.exists(PINN1_name): # Loading 
    N1.load_state_dict(torch.load(PINN1_name))
else: # Training 
    N1.train()
    N1.fit()
    torch.save(N1.state_dict(),PINN1_name)

# Results 
N1.eval()
[X1,rho1,u1,p1]=N1.Solution()

# 2 Dimension Simulation 
NL2=[2,50,20,50,4]
N2=PINN_2DFlow(NL2,Dist,G)
if RETRAIN==False and os.path.exists(PINN2_name):# Loading
    N2.load_state_dict(torch.load(PINN2_name))
else: # Training 
    N2.train()
    N2.fit()
    torch.save(N2.state_dict(),PINN2_name)

# # Results
N2.eval()
# Some interior points results 
X , rho , u , v , p , vmod  , theta = N2.Solution(X) 

# ONE DIMENSIONAL RESULTS 
x=torch.linspace(N2.Distance.Ix[0],N2.Distance.Ix[1],100).reshape(100,1)
M=torch.zeros_like(x)
u_mean=torch.zeros_like(x)
rho_mean=torch.zeros_like(x)
p_mean=torch.zeros_like(x)
for i , xc in enumerate(x):
    rho_mean[i],u_mean[i],p_mean[i],M[i]=N2.One_Dimensional_results(xc)

# Mass flow figure 
fig=plt.figure()
plt.xlabel(r'$x$')
plt.ylabel(r'$\dot{m}$')
plt.ylim(0.5, 1.5)
with torch.no_grad():
    plt.plot(x,M,label='2D model')
    plt.plot(X1,rho1*u1*N1.Area_variation(X1),'--',label='1D model')
    plt.plot(x,0*x+N1.Area_variation(torch.tensor(N1.Ix[0])),label='Constant',
             linestyle='dashdot')
plt.legend()
fig.savefig(folder_svg+'Mass flow.svg', format='svg')
fig.savefig(folder_pgf+'Mass flow.pgf',dpi=1000)
plt.close() 

fig=plt.figure()
plt.xlabel(r'$x$')
plt.ylabel(r'$\bar{u}$')
with torch.no_grad():
    plt.plot(x,u_mean,label='2D model')
    plt.plot(X1,u1,'--',label='1D model')
plt.legend()
fig.savefig(folder_svg+'Velocity comparison.svg', format='svg')
fig.savefig(folder_pgf+'Velocity comparison.pgf',dpi=1000)
plt.close()

fig=plt.figure()
plt.xlabel(r'$x$')
plt.ylabel(r'$\bar{p}$')
with torch.no_grad():
    plt.plot(x,p_mean,label='2D model')
    plt.plot(X1,p1,'--',label='1D model')
plt.legend()
fig.savefig(folder_svg+'Pressure comparison.svg', format='svg')
fig.savefig(folder_pgf+'Pressure comparison.pgf', dpi=1000)
plt.close() 

fig=plt.figure()
plt.xlabel(r'$x$')
plt.ylabel(r'$\bar\rho$')
with torch.no_grad():
    plt.plot(x,rho_mean,label='2D model')
    plt.plot(X1,rho1,'--',label='1D model')
plt.legend()
fig.savefig(folder_svg+'Density comparison.svg', format='svg')
fig.savefig(folder_pgf+'Density comparison.pgf', dpi=1000)
plt.close()

# TWO DIMENSIONAL RESULTS 
# DENSITY
Dom.figure(lambda X : N2.Solution(X)[1],r'$\rho(\mathbf{x})$','Density 2D.pgf',folder_pgf)
os.remove(folder_pgf+'Density 2D-img1.png')
plt.close()

# PRESSURE    
Dom.figure(lambda X : N2.Solution(X)[4],r'$p(\mathbf{x})$','Pressure 2D.pgf',folder_pgf)
os.remove(folder_pgf+'Pressure 2D-img1.png')
plt.close()

# VELOCITY MODULUS
Dom.figure(lambda X : N2.Solution(X)[5],r'$||\mathbf{v}(\mathbf{x})||_2$',
            'Velocity modulus.pgf',folder_pgf)
os.remove(folder_pgf+'Velocity modulus-img1.png')
plt.close()
       
# VELOCITY 
X , rho , u , v , p , vmod , theta = N2.Solution(N=300) # Some interior points results 
fig=plt.figure()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
with torch.no_grad():
    norm= mcolors.Normalize(vmin=0,vmax=torch.max(vmod))
    plt.quiver(X[:,0],X[:,1],u,v,vmod,cmap='jet',norm=norm,
                scale_units='xy',scale=0.75, width=0.003, 
                headwidth=4)
    x=torch.linspace(N2.Distance.Ix[0],N2.Distance.Ix[1],100)
    plt.plot(x,N2.Distance.f1(x),color='black',label=r'$\partial\Omega$')
    plt.plot(x,N2.Distance.f2(x),color='black')
cbar=plt.colorbar()
cbar.set_label(r'$||\mathbf{v}(\mathbf{x})||_2$')
plt.legend()
fig.savefig(folder_pgf+'Velocity field.pgf', dpi=1000) 
fig.savefig(folder_svg+'Velocity field.svg', format='svg')
plt.close()