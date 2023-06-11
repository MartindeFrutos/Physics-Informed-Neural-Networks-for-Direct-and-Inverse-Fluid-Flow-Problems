# 2 Dim Flow Forward problem. Solving Euler equations in a 2D complex domain. 
import torch 
import math as m 
from tqdm import tqdm 
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

## PINN that solves a attached shock wave 
class PINN_Shockwave(torch.nn.Module):
    def __init__(self,NL,Distance,Boundary,folder=None):
        # Inheritance from torch.nn.Module initialization
        super(PINN_Shockwave,self).__init__()
        
        # Store the NN structure 
        self.NL=NL
        
        # Initilize layers
        for i in range(0,len(NL)-2):
            setattr(self, f'layer{i}', torch.nn.Linear(NL[i],NL[i+1]))
            # setattr(self,f'batchnorm{i}', torch.nn.BatchNorm1d(NL[i+1]))
        self.layer_y1 = torch.nn.Linear(self.NL[-2], 1) # Associanted to rho 
        self.layer_y2 = torch.nn.Linear(self.NL[-2], 1) # Associated to |v|
        self.layer_y3 = torch.nn.Linear(self.NL[-2], 1) # Associated to theta
        self.layer_y4 = torch.nn.Linear(self.NL[-2], 1) # Associated to p 
        
        # Initialize activation function 
        self.activation_function=torch.nn.Sigmoid()
        
        # Some parameters for training 
        self.lr=0.01
        self.max_iter=10000 
        self.change_lr=self.max_iter/3
        self.batch_size=500  
        self.eps2=25
        self.GDW=0.0
        
        # Parameters of the equation
        self.gamma=1.4
        
        # Geometry conditions 
        self.delta=10.6229*(m.pi)/180
        self.beta=40*(m.pi)/180
        
        # Entry conditions 
        self.M1=2
        self.u1=1
        self.rho1=1
        self.p1=(self.u1/self.M1)**2*self.rho1/self.gamma
        
        # Distance and boundary already trained
        self.Distance=Distance
        self.Boundary=Boundary
        
        # We cancel the grad calculations of the parameters of this NN so 
        # they do not change during training 
        for param in self.Distance.parameters() :
            param.requires_grad = False 
            
        for param in self.Boundary.parameters() :
            param.requires_grad = False 
        
        # Tensorboard writer
        if folder==None:
            self.writer=None
        else:
            self.writer=SummaryWriter('C:/Users/marti/Desktop/MUMAC/TFM/2 Dim Flow/'+folder+'logs')    
    
    # Process to obtain the NN output
    def forward(self,x):
        for i in range(0,len(self.NL)-2):
            x = getattr(self, f'layer{i}')(x) 
            # x = getattr(self,f'batchnorm{i}')(x)     
            x = self.activation_function(x)
        # We add a different last layer for each output 
        y1=self.layer_y1(x) # Associated to rho
        y2=self.layer_y2(x) # Associated to |v| 
        y3=self.layer_y3(x) # Associated to theta
        y4=self.layer_y4(x) # Associated p
        
        return y1 , y2, y3 , y4 
    
    # Distance function 
    def Smooth_distance(self,X):
        D1 = X[:,0].reshape(X.shape[0],1) 
        D2 = X[:,0].reshape(X.shape[0],1)
        D3 = self.Distance.forward(X)
        D4 = X[:,0].reshape(X.shape[0],1)
        return D1 , D2 ,D3 , D4 
        
    # Boundary function 
    def Boundary_extension(self,X):
        G1 = self.rho1*torch.ones(X.shape[0],1)
        G2 = self.u1*torch.ones(X.shape[0],1)
        G3 = self.Boundary.forward(X) 
        G4 = self.p1*torch.ones(X.shape[0],1) 
        return G1 , G2 , G3 , G4 
       
   # Function that computes the fluid variables and its derivatives
    def Fluid_Variables(self,X):
        
        # Compute the output of the model 
        y1,y2,y3,y4=self.forward(X)
        
        # We compute the distance and boundary functions 
        G1 , G2 , G3 , G4 = self.Boundary_extension(X)
        D1 , D2 , D3 , D4 =self.Smooth_distance(X)
        
        # We compute flow variables 
        rho=G1+D1*y1
        vmod=G2+D2*y2
        theta=G3+D3*y3
        u=(G2+D2*y2)*torch.cos(G3+D3*y3) # Horizontal component 
        v=(G2+D2*y2)*torch.sin(G3+D3*y3) # Vertical component 
        p=G4+D4*y4
             
        return rho , u , v , p , vmod , theta 
    
    # Loss function, Conservation equations (Euler = inviscid)
    def loss_function(self,k=None,verbose=False):
        
        # We define the points where we are evaluating the loss
        X=self.Distance.Construct_points(self.batch_size) 
        x=X[:,0].reshape(self.batch_size,1) ; x.requires_grad=True
        y=X[:,1].reshape(self.batch_size,1) ; y.requires_grad=True
        
        # Fluid flow ad derivatives
        rho,u,v,p,_,_ = self.Fluid_Variables(torch.cat((x,y),dim=1)) 
        
        # Internal Energy of the system 
        E = 0.5*rho*(torch.pow(u,2)+torch.pow(v,2))+p/(self.gamma-1)
        
        # Fluxes
        F1=rho*u
        F2=rho*torch.pow(u,2)+p
        F3=rho*u*v
        F4=u*(E+p)

        F1_x=torch.autograd.grad(F1,x,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        F2_x=torch.autograd.grad(F2,x,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        F3_x=torch.autograd.grad(F3,x,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        F4_x=torch.autograd.grad(F4,x,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        
        G1=rho*v
        G2=rho*u*v
        G3=rho*torch.pow(v,2)
        G4=v*(E+p)
        
        G1_y=torch.autograd.grad(G1,y,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        G2_y=torch.autograd.grad(G2,y,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        G3_y=torch.autograd.grad(G3,y,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        G4_y=torch.autograd.grad(G4,y,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        
        # Gradient Dependent Weight
        u_x=torch.autograd.grad(u,x,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        v_y=torch.autograd.grad(v,y,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        velocity_divergence=u_x+v_y
        self.GDW=1/(self.eps2*(abs(velocity_divergence)-velocity_divergence)+1)
        
        # Conservation of mass
        L1 = (1/self.batch_size)*sum(self.GDW*(F1_x+G1_y)**2)
        
        # Conservation of x-momentum 
        L2 = (1/self.batch_size)*sum(self.GDW*(F2_x+G2_y)**2)
        
        # Conservation of y-momentum  
        L3 = (1/self.batch_size)*sum(self.GDW*(F3_x+G3_y)**2)
        
        # Conservation of energy 
        L4 = (1/self.batch_size)*sum(self.GDW*(F4_x+G4_y)**2)
        
        # Total form of the loss function 
        L=L1+L2+L3+L4
        
        # Split of the loss function 
        if verbose: 
            print(f'Total Loss : {L}')
            print(f'Conservation of mass loss : {L1}')
            print(f'Conservation of momentum x loss : {L2}')
            print(f'Conservation of momentum y loss : {L3}')
            print(f'Conservation of energy loss : {L4}')
        
        # We write the loss function in tensorboard
        if torch.utils.tensorboard.writer.SummaryWriter==type(self.writer) and type(k)==int:
            self.writer.add_scalar('L' , L , global_step=k)
            self.writer.add_scalar('L1', L1, global_step=k)
            self.writer.add_scalar('L2', L2, global_step=k)
            self.writer.add_scalar('L3', L3, global_step=k)
            self.writer.add_scalar('L4', L4, global_step=k)
            
        return L 
    
    # Train the NN with some dataset
    def fit(self): 
        # We define the optimizer, adding the extra parameter 
        optimizer=torch.optim.Adam(self.parameters(),lr=self.lr)
        scheduler=StepLR(optimizer,step_size=self.change_lr,gamma=0.1) 
        for k in tqdm(range(0,self.max_iter),desc='PINN'):
            # We reset the gradient of parameters
            optimizer.zero_grad()
            loss=self.loss_function(k=k)
            # Calculate backpropagation 
            loss.backward() 
            # We do one step of optimization 
            optimizer.step() 
            scheduler.step( )
            # We print how the loss is evolving 
            if (k/(self.max_iter/5)==k//(self.max_iter/5)):
                print(f'\nPINN iteration : {k}. Loss function : {float(loss.item())}.')
                
    # Obtain the results after training 
    def Solution(self,X=None,N=2000):
        
        # If we do not choose especific points we pick random
        if X==None : X=self.Distance.Construct_points(N)
        
        # Computing the fluid variables in that points 
        rho,u,v,p,vmod,theta = self.Fluid_Variables(X) 
        
        return X , rho , u , v , p , vmod , theta 
    
    # Mass flux at a certain slice x 
    def One_Dimensional_results(self,x): 
        xp=torch.ones(self.batch_size,1)*x
        yp=torch.linspace(float(self.Distance.f2(x)),
                          float(self.Distance.f1(x)),
                          self.batch_size).reshape(self.batch_size,1)
        X=torch.cat((xp,yp),dim=1)
        X , rho , u , _ , p , _ , _ = self.Solution(X)
        mass_flux=torch.trapz(rho*u,yp,dim=0)
        Ly=(float(self.Distance.f1(x))-
            float(self.Distance.f2(x)))
        rho1=torch.trapz(rho,yp,dim=0)/Ly
        u1=torch.trapz(u,yp,dim=0)/Ly
        p1=torch.trapz(p,yp,dim=0)/Ly 
        return  rho1, u1 , p1 , mass_flux 
    
# PINN that solves the Euler fluid equations on two dimensions
class PINN_2DFlow(torch.nn.Module):
    def __init__(self,NL,Distance,Boundary,folder=None,normalization=None):
        # Inheritance from torch.nn.Module initialization
        super(PINN_2DFlow,self).__init__()
        
        # Store the NN structure 
        self.NL=NL
        
        # Initilize layers
        for i in range(0,len(NL)-2):
            setattr(self, f'layer{i}', torch.nn.Linear(NL[i],NL[i+1]))
            # setattr(self,f'batchnorm{i}', torch.nn.BatchNorm1d(NL[i+1]))
        self.layer_y1 = torch.nn.Linear(self.NL[-2], 1) # Associanted to rho 
        self.layer_y2 = torch.nn.Linear(self.NL[-2], 1) # Associated to |v|
        self.layer_y3 = torch.nn.Linear(self.NL[-2], 1) # Associated to theta
        self.layer_y4 = torch.nn.Linear(self.NL[-2], 1) # Associated to p 
        
        # Initialize activation function 
        self.activation_function=torch.nn.Sigmoid()
        
        # Some parameters for training 
        self.lr=0.01
        self.max_iter=10000 
        self.change_lr=self.max_iter/5
        self.batch_size=500  
        
        # Parameters of the equation
        self.gamma=1.4
        
        # Entry conditions 
        self.RHO0=1.0
        self.VMOD0=1.0 # We assume horizontal flow at the entry 
        self.P0=1.0
        
        # Distance and boundary already trained
        self.Distance=Distance
        self.Boundary=Boundary
        
        # We cancel the grad calculations of the parameters of this NN so 
        # they do not change during training 
        for param in self.Distance.parameters() :
            param.requires_grad = False 
            
        for param in self.Boundary.parameters() :
            param.requires_grad = False 
        
        # Tensorboard writer
        if folder==None:
            self.writer=None
        else:
            self.writer=SummaryWriter('C:/Users/marti/Desktop/MUMAC/TFM/2 Dim Flow/'+folder+'logs')    
        
        # Flag to determine wether normalize all losses terms 
        self.normalization=normalization
        if self.normalization==True:
            for i in range(1,5):
                setattr(self,f'NormL{i}',1.0)
    
    # Process to obtain the NN output
    def forward(self,x):
        for i in range(0,len(self.NL)-2):
            x = getattr(self, f'layer{i}')(x) 
            # x = getattr(self,f'batchnorm{i}')(x)     
            x = self.activation_function(x)
        # We add a different last layer for each output 
        y1=self.layer_y1(x) # Associated to rho
        y2=self.layer_y2(x) # Associated to |v| 
        y3=self.layer_y3(x) # Associated to theta
        y4=self.layer_y4(x) # Associated p
        
        return y1 , y2, y3 , y4 
    
    # Distance function 
    def Smooth_distance(self,X):
        D1 = X[:,0].reshape(X.shape[0],1) 
        D2 = X[:,0].reshape(X.shape[0],1)
        D3 = self.Distance.forward(X)
        D4 = X[:,0].reshape(X.shape[0],1)
        return D1 , D2 ,D3 , D4 
        
    # Boundary function 
    def Boundary_extension(self,X):
        G1 = self.RHO0*torch.ones(X.shape[0],1)
        G2 = self.VMOD0*torch.ones(X.shape[0],1)
        G3 = self.Boundary.forward(X) 
        G4 = self.P0*torch.ones(X.shape[0],1) 
        return G1 , G2 , G3 , G4 
       
   # Function that computes the fluid variables and its derivatives
    def Fluid_Variables(self,X):
        
        # Compute the output of the model 
        y1,y2,y3,y4=self.forward(X)
        
        # We compute the distance and boundary functions 
        G1 , G2 , G3 , G4 = self.Boundary_extension(X)
        D1 , D2 , D3 , D4 =self.Smooth_distance(X)
        
        # We compute flow variables 
        rho=G1+D1*y1
        vmod=G2+D2*y2
        theta=G3+D3*y3
        u=(G2+D2*y2)*torch.cos(G3+D3*y3) # Horizontal component 
        v=(G2+D2*y2)*torch.sin(G3+D3*y3) # Vertical component 
        p=G4+D4*y4
             
        return rho , u , v , p , vmod , theta 
    
    # Loss function, Conservation equations (Euler = inviscid)
    def loss_function(self,k=None,verbose=False):
        
        # We define the points where we are evaluating the loss
        X=self.Distance.Construct_points(self.batch_size) 
        x=X[:,0].reshape(self.batch_size,1) ; x.requires_grad=True
        y=X[:,1].reshape(self.batch_size,1) ; y.requires_grad=True
        
        # Fluid flow ad derivatives
        rho,u,v,p,_,_ = self.Fluid_Variables(torch.cat((x,y),dim=1)) 
        
        # Internal Energy of the system 
        E = 0.5*rho*(torch.pow(u,2)+torch.pow(v,2))+p/(self.gamma-1)
        
        # Fluxes
        F1=rho*u
        F2=rho*torch.pow(u,2)+p
        F3=rho*u*v
        F4=u*(E+p)

        F1_x=torch.autograd.grad(F1,x,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        F2_x=torch.autograd.grad(F2,x,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        F3_x=torch.autograd.grad(F3,x,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        F4_x=torch.autograd.grad(F4,x,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        
        G1=rho*v
        G2=rho*u*v
        G3=rho*torch.pow(v,2)
        G4=v*(E+p)
        
        G1_y=torch.autograd.grad(G1,y,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        G2_y=torch.autograd.grad(G2,y,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        G3_y=torch.autograd.grad(G3,y,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        G4_y=torch.autograd.grad(G4,y,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        
        # Conservation of mass
        L1 = (1/self.batch_size)*sum((F1_x+G1_y)**2)
        
        # Conservation of x-momentum 
        L2 = (1/self.batch_size)*sum((F2_x+G2_y)**2)
        
        # Conservation of y-momentum  
        L3 = (1/self.batch_size)*sum((F3_x+G3_y)**2)
        
        # Conservation of energy 
        L4 = (1/self.batch_size)*sum((F4_x+G4_y)**2)
        
        # Total form of the loss function 
        L=L1+L2+L3+L4
        
        # Split of the loss function 
        if verbose: 
            print(f'Total Loss : {L}')
            print(f'Conservation of mass loss : {L1}')
            print(f'Conservation of momentum x loss : {L2}')
            print(f'Conservation of momentum y loss : {L3}')
            print(f'Conservation of energy loss : {L4}')
            
        return L 

        # Fluxes
        F1=rho*u
        F2=rho*torch.pow(u,2)+p
        F3=rho*u*v
        F4=u*(E+p)
        
        F1_x=torch.autograd.grad(F1,x,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        F2_x=torch.autograd.grad(F2,x,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        F3_x=torch.autograd.grad(F3,x,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        F4_x=torch.autograd.grad(F4,x,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        
        G1=rho*v
        G2=rho*u*v
        G3=rho*torch.pow(v,2)
        G4=v*(E+p)
        
        G1_y=torch.autograd.grad(G1,y,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        G2_y=torch.autograd.grad(G2,y,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        G3_y=torch.autograd.grad(G3,y,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        G4_y=torch.autograd.grad(G4,y,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        
        # Conservation of mass
        L1 = sum((F1_x+G1_y)**2)
        
        # Conservation of x-momentum 
        L2 = sum((F2_x+G2_y)**2)
        
        # Conservation of y-momentum  
        L3 = sum((F3_x+G3_y)**2)
        
        # Conservation of energy 
        L4 = sum((F4_x+G4_y)**2)
         
        # Normalization of the loss 
        if type(k)==int and k==0 and self.normalization:
            self.NormL1=float(L1)
            self.NormL2=float(L2)
            self.NormL3=float(L3)
            self.NormL4=float(L4)
        
        if self.normalization:
            L1=L1/self.NormL1
            L2=L2/self.NormL2
            L3=L3/self.NormL3
            L4=L4/self.NormL4
            
        # Split of the loss function 
        if verbose: 
            print(f'Conservation of mass loss : {L1}')
            print(f'Conservation of momentum x loss : {L2}')
            print(f'Conservation of momentum y loss : {L3}')
            print(f'Conservation of energy loss : {L4}')
            
        # We write the loss function in tensorboard
        if self.writer==None or self.k==None:
            pass 
        else: 
            self.writer.add_scalar('L1', L1, global_step=k)
            self.writer.add_scalar('L2', L2, global_step=k)
            self.writer.add_scalar('L3', L3, global_step=k)
            self.writer.add_scalar('L4', L4, global_step=k)
            
        return L1+L2+L3+L4
        
    # Train the NN with some dataset
    def fit(self): 
        # We define the optimizer (Adam or SGD)
        optimizer=torch.optim.Adam(self.parameters(),lr=self.lr)
        scheduler=StepLR(optimizer,step_size=self.change_lr,gamma=0.1) 
        for k in tqdm(range(0,self.max_iter),desc='PINN'):
            # We reset the gradient of parameters
            optimizer.zero_grad()
            loss=self.loss_function(k=k)
            # Calculate backpropagation 
            loss.backward() 
            # We do one step of optimization 
            optimizer.step() 
            scheduler.step()
            # We print how the loss is evolving 
            if (k/(self.max_iter/5)==k//(self.max_iter/5)):
                print(f'\nPINN iteration : {k}. Loss function : {loss.item()}.')            
    
    # Obtain the results after training 
    def Solution(self,X=None,N=2000):
        
        # If we do not choose especific points we pick random
        if X==None : X=self.Distance.Construct_points(N)
        
        # Computing the fluid variables in that points 
        rho,u,v,p,vmod,theta = self.Fluid_Variables(X) 
        
        return X , rho , u , v , p , vmod , theta 
    
    # Mass flux at a certain slice x 
    def One_Dimensional_results(self,x): 
        xp=torch.ones(self.batch_size,1)*x
        yp=torch.linspace(float(self.Distance.f2(x)),
                          float(self.Distance.f1(x)),
                          self.batch_size).reshape(self.batch_size,1)
        X=torch.cat((xp,yp),dim=1)
        X , rho , u , _ , p , _ , _ = self.Solution(X)
        mass_flux=torch.trapz(rho*u,yp,dim=0)
        Ly=(float(self.Distance.f1(x))-
            float(self.Distance.f2(x)))
        rho1=torch.trapz(rho,yp,dim=0)/Ly
        u1=torch.trapz(u,yp,dim=0)/Ly
        p1=torch.trapz(p,yp,dim=0)/Ly 
        return  rho1, u1 , p1 , mass_flux 
    
   
# PINN that solves the one dimensional simplification 
# under slow variation of area A(x)
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
        
        # Initialize activation function 
        self.activation_function=torch.nn.Sigmoid()
        
        # Some parameters for training 
        self.lr=0.01
        self.max_iter=5000
        
        # Parameters of the equation
        self.gamma=1.4
        self.A0=1
        
        # Boundary conditions (Dirichlet for now)
        self.rho_inf=1
        self.u_inf=1
        self.p_inf=1
        
        # Domain parameters 
        self.Ix=[0,4]
        self.batch_size=50 # batch_size
        
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
        return y1 , y2, y3  
    
    # Distance function 
    def Smooth_distance(self,X):
        return X
        
    # Boundary function 
    def Boundary_extension(self,X):
        return self.U0[0] , self.U0[1] , self.U0[2]
    
    # Area variation 
    def Area_variation(self,X):
        # The Area follows a general logistic equation
        AL=self.A0*2 # Final value of the area 
        k=2
        
        # related with the curvature 
        return self.A0+AL/(1+torch.exp(-k*(X-self.Ix[1]*0.5-self.Ix[0]*0.5)))
        # return  2*(2-1/(1+torch.exp(-(X-0.5))))
   # Function that computes the fluid variables and its derivatives
    def Fluid_Variables_Derivatives(self,X):
        
        # Compute the output of the model 
        y1,y2,y3=self.forward(X)
        
        # We compute the distance and boundary functions 
        G1, G2, G3 = self.Boundary_extension(X)
        D=self.Smooth_distance(X)
        
        # We compute flow variables 
        rho=G1+D*y1
        u=G2+D*y2
        p=G3+D*y3
        
        # We compute the derivatives of the flow variables 
        rho_x=torch.autograd.grad(rho,X,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        u_x=torch.autograd.grad(u,X,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        p_x=torch.autograd.grad(p,X,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
                
        return rho , u , p , rho_x , u_x , p_x 
    # Loss function, depends on the EDO to solve
    def loss_function(self,k=None):
        # We define the points where we are evaluating the loss
        X=self.Ix[0]+(self.Ix[1]-self.Ix[0])*torch.rand((self.batch_size,1))
        X.requires_grad = True
        
        # Fluid flow ad derivatives
        rho , u , p , rho_x , u_x , p_x  = self.Fluid_Variables_Derivatives(X) 
        
        # Area variation
        A = self.Area_variation(X)
        A_x = torch.autograd.grad(A,X,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        
        # Internal Energy of the system 
        E = 0.5*rho*torch.pow(u,2)+p/(self.gamma-1)
        E_x = 0.5*rho_x*torch.pow(u,2)+rho*u*u_x+p_x/(self.gamma-1)

        # Flows 
        F1x = A*(rho_x*u+rho*u_x)
        F2x = A*(rho_x*torch.pow(u,2)+2*rho*u*u_x + p_x)
        F3x = A*(u_x*(E+p)+u*(E_x+p_x))
        
        # Source terms
        S1 = A_x*rho*u
        S2 = A_x*rho*torch.pow(u,2)
        S3 = A_x*u*(E+p)
        
        # Conservation of mass
        L1 = 1/self.batch_size*sum((F1x+S1)**2)
        
        # Conservation of momentum 
        L2 = 1/self.batch_size*sum((F2x+S2)**2)
         
        # Conservation of energy 
        L3 = 1/self.batch_size*sum((F3x+S3)**2)
        
        return L1+L2+L3
        
    # Train the NN with some dataset
    def fit(self): 
        # We define the optimizer (Adam or SGD)
        optimizer=torch.optim.Adam(self.parameters(),lr=self.lr)
        for k in tqdm(range(0,self.max_iter),desc='1D PINN loss'):
            # We reset the gradient of parameters
            optimizer.zero_grad()
            loss=self.loss_function(k)
            # Calculate backpropagation 
            loss.backward() 
            # We do one step of optimization 
            optimizer.step() 
            if (k/(self.max_iter/5)==k//(self.max_iter/5)):
                print(f'\n1D Loss function : {loss.item()}.')            
    
    # Obtain the results after training 
    def Solution(self):
        # Grid 
        x_coordinates=torch.linspace(self.Ix[0],self.Ix[1],100)
        X=x_coordinates.reshape(100,1)
        
        # NN output 
        y1,y2,y3=self.forward(X)
        
        # We compute the distance and boundary functions 
        G1, G2, G3 = self.Boundary_extension(X)
        D=self.Smooth_distance(X)

        # We compute flow variables 
        rho=G1+D*y1
        u=G2+D*y2
        p=G3+D*y3
        
        return X , rho , u , p