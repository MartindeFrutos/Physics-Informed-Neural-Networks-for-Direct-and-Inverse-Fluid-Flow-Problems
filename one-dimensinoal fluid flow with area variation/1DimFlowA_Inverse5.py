# 1 Dimensional Flow with Area Variation
# General Inverse Problem or Design Problem
# We set initial conditions, velocity at the extremum and flux. 
# Find for area distribution that accomplish all that and also has minimum volume. 
import torch 
from tqdm import tqdm 
import pandas as pd
import matplotlib  
import matplotlib.pyplot as plt 
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size': 13
})
writer = SummaryWriter('C:/Users/marti/Desktop/MUMAC/TFM/1 Dim Flow with Area Variation/Inverse5/logs')    
class PINN_1DFlowA(torch.nn.Module):
    def __init__(self,NL):
        
        # Inheritance from torch.nn.Module initialization
        super(PINN_1DFlowA,self).__init__()
        
        # Store the NN structure 
        self.NL=NL
        
        # Initilize layers
        for i in range(0,len(NL)-2):
            setattr(self, f'layer{i}', torch.nn.Linear(NL[i],NL[i+1]))
            # setattr(self,f'batchnorm{i}', torch.nn.BatchNorm1d(NL[i+1]))
        self.layer_y1 = torch.nn.Linear(self.NL[-2], 1)
        self.layer_y2 = torch.nn.Linear(self.NL[-2], 1)
        self.layer_y3 = torch.nn.Linear(self.NL[-2], 1)
        self.layer_y4 = torch.nn.Linear(self.NL[-2], 1)
        
        # Initialize activation function 
        self.activation_function=torch.nn.Sigmoid()
        
        # Some parameters for training 
        self.lr=0.01
        self.max_iter=10000
        self.change_lr=self.max_iter/5
        
        # Parameters of the equation
        self.gamma=1.4
        
        # Regularizers 
        self.beta=0.1
        
        # Boundary conditions 
        self.RHO0=1
        self.U0=1
        self.P0=1
        
        self.G=torch.tensor([1.0]) # Mass flux
        
        self.RHOL=torch.tensor([1.3])
        self.UL=torch.tensor([0.5]) # Velocity at the extremum 
        self.PL=torch.tensor([1.5]) # Pressure at the extremum
      
        # Domain parameters 
        self.Ix=[0.0,1.0]
        self.batch_size=10 # batch_size
        
    # Process to obtain the NN output
    def forward(self,x):
        for i in range(0,len(self.NL)-2):
            x = getattr(self, f'layer{i}')(x) 
            # x = getattr(self,f'batchnorm{i}')(x)     
            x = self.activation_function(x)
        # We add a different last layer for each output 
        y1=self.layer_y1(x) # Associated to rho
        y2=self.layer_y2(x) # Associated to u 
        y3=self.layer_y3(x) # Associated to p
        y4=self.layer_y4(x) # Area variation A(x)
        
        return y1 , y2, y3 , y4 
    
    # Distance function 
    def Smooth_distance(self,X):
        # One boundary condition 
        D1 = X
        # Two boundary conditions 
        D2 = (X-self.Ix[0])*(self.Ix[1]-X)
        return D1 , D2  
        
    # Boundary function 
    def Boundary_extension(self,X):
        A0=self.G/(self.RHO0*self.U0)
        G1=self.RHO0 
        G2=self.U0 + (X-self.Ix[0])*(self.UL-self.U0)/(self.Ix[1]-self.Ix[0])
        G3=self.P0
        G4=A0 
        return G1 , G2 , G3 , G4 
       
   # Function that computes the fluid variables and its derivatives
    def Fluid_Variables(self,X):
        
        # Compute the output of the model 
        y1,y2,y3,y4=self.forward(X)
        
        # We compute the distance and boundary functions 
        G1, G2, G3 , G4 = self.Boundary_extension(X)
        D1 ,D2 = self.Smooth_distance(X)
        
        # We compute flow variables 
        rho=G1+D1*y1
        u=G2+D2*y2
        p=G3+D1*y3
        A=G4+D1*y4
             
        return rho , u , p , A
    # Loss function, depends on the EDO to solve
    def loss_function(self,k=None,verbose=False):
        # We define the points where we are evaluating the loss
        X = self.Ix[0]+(self.Ix[1]-self.Ix[0])*torch.rand((self.batch_size,1))
        X.requires_grad = True
        
        # Fluid flow ad derivatives
        rho,u,p,A = self.Fluid_Variables(X) 
        A_x=torch.autograd.grad(A,X,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        
        # Internal Energy of the system 
        E = 0.5*rho*torch.pow(u,2)+p/(self.gamma-1)

        # Flows 
        F1=rho*u
        F2=rho*torch.pow(u,2)+p
        F3=u*(E+p)
        
        F1_x=torch.autograd.grad(F1,X,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        F2_x=torch.autograd.grad(F2,X,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        F3_x=torch.autograd.grad(F3,X,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        
        # Source terms
        S1 = A_x*F1
        S2 = A_x*rho*torch.pow(u,2)
        S3 = A_x*F3
        
        # Conservation of mass
        L1 = sum((A*F1_x+S1)**2)
        writer.add_scalar('L1', L1, global_step=k)
        
        # Conservation of momentum 
        L2 = sum((A*F2_x+S2)**2)
        writer.add_scalar('L2', L2, global_step=k)
        
        # Conservation of energy 
        L3 = sum((A*F3_x+S3)**2)
        writer.add_scalar('L3', L3, global_step=k)
        
        # Volume regularizer 
        LV = self.beta*abs(torch.trapz(A.reshape(self.batch_size),X.reshape(self.batch_size)))
        writer.add_scalar('LV', LV, global_step=k)
        
        if verbose: 
            print(L1,L2,L3,LV)
            
        return L1+L2+L3+LV
        
    # Train the NN with some dataset
    def fit(self): 
        # We define the optimizer (Adam or SGD)
        optimizer=torch.optim.Adam(self.parameters(),lr=self.lr)
        scheduler=StepLR(optimizer,step_size=self.change_lr,gamma=0.1) 
        for k in tqdm(range(0,self.max_iter),desc='Loss'):
            # We reset the gradient of parameters
            optimizer.zero_grad()
            loss=self.loss_function(k)
            # Calculate backpropagation 
            loss.backward() 
            writer.add_scalar('Global Loss', loss.item(), global_step=k)
            # We do one step of optimization 
            optimizer.step() 
            scheduler.step() 
            # if (k/(self.max_iter/20)==k//(self.max_iter/20)):
                 # print(f'Iteration : {k}. Loss function : {loss.item()}.')            
    
    # Obtain the results after training 
    def Solution(self):
        # Grid 
        x_coordinates = torch.linspace(self.Ix[0],self.Ix[1],100)
        X = x_coordinates.reshape(100,1)
        
        rho,u,p,A = self.Fluid_Variables(X) 
        
        return X , rho , u , p , A

# Path 
folder='Inverse5/'

# Beta influence 
beta=[0.01,0.1,1,2,5,10]
maxiter=[2000,2500,2500,5000,10000,10000]
A=torch.zeros(100,len(beta))

for i,b in enumerate(beta):
    print(f'Computing solution for beta={b}')
    N=PINN_1DFlowA([1,20,20,20,4])
    N.max_iter=maxiter[i]; N.beta=b
    N.train()
    N.fit() # we actually perfom the fitting of the model    
    N.loss_function(verbose=True)
    N.eval()
    [X,_,_,_,a]=N.Solution()
    A[:,i]=a.reshape(100)

plt.figure()
plt.xlabel('$x$')
plt.ylabel('$A$')
with torch.no_grad():
    plt.plot(X,A[:,0],label='$\\beta=0.01$')
    plt.plot(X,A[:,1],label='$\\beta=0.1$')
    plt.plot(X,A[:,2],label='$\\beta=1$')
    plt.plot(X,A[:,3],label='$\\beta=2$')
    plt.plot(X,A[:,4],label='$\\beta=5$')
    plt.plot(X,A[:,5],label='$\\beta=10$')
plt.legend()
plt.tight_layout()
plt.savefig(folder+'beta influence.svg',format='svg')
plt.savefig(folder+'beta influence.pgf',dpi=1000)


# For the desired beta 
N=PINN_1DFlowA([1,20,20,20,4])
N.train()
N.fit()

# Results 
N.eval()
[X,rho,u,p,A]=N.Solution()
plt.figure()
plt.xlabel(r'$x$')
with torch.no_grad():
    plt.plot(X,u,label=r'$u$')
    plt.plot(X,rho,label=r'$\rho$')
    plt.plot(X,p,label=r'$p$')
    plt.plot(X,rho*u*A,label=r'$\dot m$')
plt.legend()
plt.savefig(folder+'Variables distribution full inverse.pgf',dpi=1000)
plt.savefig(folder+f'VD beta={N.beta}.svg',format='svg')

