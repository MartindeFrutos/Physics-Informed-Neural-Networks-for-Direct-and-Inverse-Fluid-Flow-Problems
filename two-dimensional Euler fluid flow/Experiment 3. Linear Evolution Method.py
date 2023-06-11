# Experiment 3. Testing Linear Evolution Method
import os 
import torch 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from Domain_classes import Domain , Boundary

mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size': 13
})

# Directories
folder='3. Linear Evolution method test data/'
tex_folder=folder+'PGF/'
boundary_name=folder+'Trained_boundary.pt'

# Do we want to retrain for some reason ? 
RETRAIN=False

# Geometry 
Ix=[0.0,4.0]
f1 = lambda x : 0.5+1.0/(1+torch.exp(-2.0*(x-Ix[1]*0.5-Ix[0]*0.5)))
f2 = lambda x : -f1(x)

Dom=Domain(Ix,f1,f2)
Bound=Boundary(0.0,Ix,f1,f2)

if RETRAIN==False and os.path.exists(boundary_name):
    Bound.load_state_dict(torch.load(boundary_name))
else:
    Bound.fit()
    torch.save(Bound.state_dict(),boundary_name)


# Can we compute G analytically 
def analytical_G(X):
    x=X[:,0] ; x.requires_grad=True 
    y=X[:,1] ; y.requires_grad=True 
    # First we compute the tangent vector
    f1x=torch.autograd.grad(f1(x),x,torch.ones_like(x),
                            retain_graph=True,create_graph=True)[0]
    f2x=torch.autograd.grad(f2(x),x,torch.ones_like(x),
                            retain_graph=True,create_graph=True)[0]
    theta1=torch.atan2(f1x,torch.ones_like(f1x))
    theta2=torch.atan2(f2x,torch.ones_like(f2x))
    
    G=theta2 + (y-f2(x))*(theta1-theta2)/(f1(x)-f2(x))
    
    return G 

# Plots 
X=Dom.Construct_points(10000)
Y=Bound.forward(X)
fig=plt.figure()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
with torch.no_grad():
    plt.scatter(X[:,0],X[:,1],c=Y)
    x=torch.linspace(Dom.Ix[0],Dom.Ix[1],100)
    plt.plot(x,Dom.f1(x)+0.03,color='black',label='boundary')
    plt.plot(x,Dom.f2(x)-0.03,color='black')
cbar=plt.colorbar()
cbar.set_label(r'$G_\varphi(\mathbf{x})$')
plt.legend()
fig.savefig(folder+'NN Boundary extension function.svg', format='svg')
plt.close()


Dom.figure(analytical_G,r'$G_\varphi(\mathbf{x})$','Analytical G.pgf',tex_folder)
os.remove(tex_folder+'Analytical G-img1.png')
fig.savefig(folder+'Ruled Boundary extension function.svg', format='svg')
plt.close()

Y2=analytical_G(X)

fig=plt.figure()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
with torch.no_grad():
    plt.scatter(X[:,0],X[:,1],c=Y2)
    x=torch.linspace(Dom.Ix[0],Dom.Ix[1],100)
    plt.plot(x,Dom.f1(x)+0.03,color='black',label='boundary')
    plt.plot(x,Dom.f2(x)-0.03,color='black')
cbar=plt.colorbar()
cbar.set_label(r'$G_\varphi(\mathbf{x})$')
plt.legend()
fig.savefig(folder+'Ruled Boundary extension function.svg', format='svg')
fig.savefig(tex_folder+'Analytical G.pgf',dpi=1000)

