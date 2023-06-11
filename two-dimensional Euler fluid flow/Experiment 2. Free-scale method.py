# Experiment 2. Testing Free-scale method
import os 
import torch 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from Domain_classes import Domain, Distance 

mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size': 13
})

# Directories
folder='2. Free-scale method test data/'
tex_folder=folder+'PGF/'
distance_name=folder+'Trained_distance.pt'

# Do we want to retrain for some reason ? 
RETRAIN=False

# Geometry 
Ix=[0.0,4.0]
f1 = lambda x : 0.5+1.0/(1+torch.exp(-2.0*(x-Ix[1]*0.5-Ix[0]*0.5)))
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

# Plots 
X=Dom.Construct_points(10000)
Y=Dist.forward(X)
plt.rcParams.update({'font.size': 12, 'text.usetex': True})
fig=plt.figure()
plt.title(r'Smooth extension of distance function')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
with torch.no_grad():
    plt.scatter(X[:,0],X[:,1],c=Y)
    plt.scatter(Dist.X3[:,0],Dist.X3[:,1],marker='x',color='red',label=r'Training points')
    x=torch.linspace(Dist.Ix[0],Dist.Ix[1],100)
    plt.plot(x,Dist.f1(x)+0.03,color='black',label='boundary')
    plt.plot(x,Dist.f2(x)-0.03,color='black')
cbar=plt.colorbar()
cbar.set_label(r'$D(\mathbf{x})$')
plt.legend()
fig.savefig(folder+'Distance.svg', format='svg')
plt.close()
# Imagine the boundary has moved to 
f12 = lambda x : 0.5+2.0/(1+torch.exp(-2.0*(x-Ix[1]*0.5-Ix[0]*0.5)))
f22 = lambda x : 0*x# -f12(x)

# To do the plot we kind of do a change of variable 
# We generate the new domain 
Dom_new=Domain(Ix,f12,f22)

# We compute the new points 
X_new=Dom_new.Construct_points(10000)

# These functions are just tests, the final versions are incorporated on the Distance class.
# We compute the reescale factor
def rescaling(x,y):
    L1=f1(x)-f2(x)
    L2=f12(x)-f22(x)
    yc1=(f1(x)+f2(x))*0.5
    yc2=(f12(x)+f22(x))*0.5
    return yc1+(y-yc2)*L1/L2 

def NewDistance(X_new):   
    # We change the coordinates to compute the distance 
    X_new_rescaled=torch.zeros_like(X_new) 
    X_new_rescaled[:,0]=X_new[:,0]
    X_new_rescaled[:,1]=rescaling(X_new[:,0],X_new[:,1])
    Y=Dist.forward(X_new_rescaled)
    return Y 

Dom_new.figure(NewDistance,r'$D_\varphi(\bar x)',
               'Distance different domain.pgf',tex_folder)
fig.savefig(folder+'Distance on different domain.svg', format='svg')
plt.close()
os.remove(tex_folder+'Distance different domain-img1.png')

