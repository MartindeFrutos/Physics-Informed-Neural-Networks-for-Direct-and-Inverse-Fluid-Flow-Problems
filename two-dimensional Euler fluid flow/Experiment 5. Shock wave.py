# Experiment 5, Shock wave formation
import os 
import math as m 
import torch  
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from Domain_classes import Domain, Distance , Boundary 
from PINN_forward_classes import PINN_Shockwave 

mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size': 13
})

# Directories
folder='5. Shock/'
folder_pgf=folder+'PGF/'
distance_name=folder+'Trained_distance.pt'
boundary_name=folder+'Trained_boundary.pt'
PINN_name=folder+'PINN.pt'

# Do we want to retrain for some reason ? 
RETRAIN=True
RETRAIN_DISTANCE=False

# Geometry 
Ix=[-2.0,2.0]

# Boundary that exactly creates the shock wave where it is supposed to
f2 = lambda x : torch.log(1+torch.exp(30*(x)))/30*(10.6229*torch.pi/180)
f1 = lambda x : f2(x-torch.cos(torch.tensor(40*torch.pi/180)))+1

# Domain pretraining
Dom=Domain(Ix,f1,f2)

# Computing the distance function 
Dist=Distance(Ix,f1,f2)
if RETRAIN_DISTANCE==False and os.path.exists(distance_name):
    Dist.load_state_dict(torch.load(distance_name))
else:
    Dist.fit()
    torch.save(Dist.state_dict(),distance_name)

# Computing the boundary function
G=Boundary(0.0,Ix,f1,f2)
if RETRAIN_DISTANCE==False and os.path.exists(boundary_name):
    G.load_state_dict(torch.load(boundary_name))
else:
    G.fit()
    torch.save(G.state_dict(),boundary_name)

# Plots 
X=Dom.Construct_points(10000)

# Distance plot 
Dom.figure(Dist.forward,r'$D_\varphi(\bar x)$','Trained distance.pgf',folder_pgf)
plt.savefig(folder+'Distance.svg',format='svg')
plt.close()
if os.path.exists(folder_pgf+'CMAP.png'):
    os.remove(folder_pgf+'Trained distance-img1.png')
else : 
    os.rename(folder_pgf+'Trained distance-img1.png',folder_pgf+'CMAP.png')

# Boundary plot 
Dom.figure(G.forward,r'$G_\varphi(\bar x)$','Boundary extension.pgf',folder_pgf)
plt.savefig(folder+'Boundary.svg',format='svg')
os.remove(folder_pgf+'Boundary extension-img1.png')
plt.close()

# 2 Dimension Simulation 
NL=[2,50,50,50,50,50,4]
N=PINN_Shockwave(NL,Dist,G,folder=folder)
N.max_iter=5000

# Theoretical solutions
M1n=N.M1*m.sin(N.beta) 
M2n=m.sqrt((2+(N.gamma-1)*M1n**2)/(2*N.gamma*M1n**2-N.gamma+1))
M2=M2n/(m.sin(N.beta-N.delta))
p2=N.p1*(2*N.gamma*M1n**2-N.gamma+1)/(N.gamma+1)
rho2=N.rho1*(N.gamma+1)*M1n**2/(2+(N.gamma-1)*M1n**2)
vmod2=M2*m.sqrt(N.gamma*p2/rho2)

# Training of the neural network 
for file in os.listdir(folder+'logs'):
    os.remove(folder+'logs/'+file)
N.fit()
N.eps2=0
# torch.save(N.state_dict(),PINN_name)
N.eval()
# Some interior points results 
X , rho , u , v , p , vmod  , theta = N.Solution(X) 

# PRESSURE    
Dom.figure(lambda X : N.Solution(X)[4],r'$p(\bar x)$','Pressure 2D eps=0.pgf',folder_pgf)
os.remove(folder_pgf+'Pressure 2D-img1.png')
plt.savefig(folder+'pressure eps=0.svg',format='svg')
plt.close()

# DENSITY 
Dom.figure(lambda X : N.Solution(X)[2],r'$\rho(\bar x)$',
            'density.pgf',folder_pgf)
plt.savefig(folder+'density.svg',format='svg')
os.remove(folder_pgf+'density-img1.png')
plt.close()

# VELOCITY MODULUS 
Dom.figure(lambda X : N.Solution(X)[5],r'$||\mathbf{v}(\bar{x})||_2$',
            'Velocity modulus.pgf',folder_pgf)
plt.savefig(folder+'velocity modulus.svg',format='svg')
os.remove(folder_pgf+'Velocity modulus-img1.png')
plt.close()