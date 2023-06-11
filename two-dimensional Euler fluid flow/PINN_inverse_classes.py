# 2 Dim Flow Inverse problem. Solving Euler equations in a 2D complex domain. 
import torch 
from tqdm import tqdm 
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from Domain_classes import VARIABLES_SAVE

# Class that solves the inverse problem for AL of the two-dimensional Euler equations.     
class PINN2D_Inverse_1parameter(torch.nn.Module):
    def __init__(self,NL,Domain,Distance,writer_folder=None,animation=False):    
        
        # Inheritance from torch.nn.Module initialization
        super(PINN2D_Inverse_1parameter,self).__init__()
        
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
        self.max_iter=5000
        self.change_lr=self.max_iter/1
        self.batch_size=250  
        
        # Parameters of the equation
        self.gamma=1.4
        
        # Entry conditions 
        self.RHO0=1.0
        self.VMOD0=1.0 # We assume horizontal flow at the entry 
        self.P0=1.0
        self.UL=0.2616 # Extra condition 
        
        # Distance and boundary already trained, initial configuration
        self.Original_Distance=Distance
        self.Domain=Domain
        
        # Parameter of the inverse problem 
        self.AL = torch.nn.Parameter(torch.tensor([1.5]))
        L=self.Domain.Ix[1]-self.Domain.Ix[0]
        self.f1 = lambda x , AL : 0.5+(AL/2-0.5)/(1+torch.exp(-2.0*(x-L/2)))
        self.f2 = lambda x , AL : -self.f1(x,AL) # We enforce symmetric behaviour 

        # We cancel the grad calculations of the parameters of this NN so 
        # they do not change during training 
        for param in self.Original_Distance.parameters() :
            param.requires_grad = False 
        
        # Tensorboard writer
        if writer_folder==None:
            self.writer=None
        else:
            self.writer=SummaryWriter('C:/Users/marti/Desktop/MUMAC/TFM/2 Dim Flow/'
                                      +writer_folder+'logs')    
            
        # Animation, write solution during training process
        self.animation=animation 
        self.VARIABLES=VARIABLES_SAVE(self.max_iter)
            
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
    
    # Rescaling function to adapt D over any domain 
    def rescale(self,X):
        
        # Variable rescaled to original domain 
        X_new=torch.zeros_like(X)
        X_new[:,0]=X[:,0]
        
        # Dimensions of the current domain
        b_up=self.f1(X[:,0],self.AL)
        b_down=self.f2(X[:,0],self.AL)
        yc=(b_up+b_down)*0.5
        
        # Dimmensions of the original domain 
        b_up_new=self.Original_Distance.f1(X[:,0])
        b_down_new=self.Original_Distance.f2(X[:,0])
        yc_new=(b_up_new+b_down_new)*0.5
        
        # Dimensionless dimension 
        eta=(X[:,1]-yc)/(b_up-yc) 
        
        # Original dimensions 
        X_new[:,1]=yc_new+eta*(b_up_new-yc_new)
        
        return X_new
    
    # Distance function 
    def Smooth_distance(self,X):
        
        # Dirichlet on entry, distance is just x 
        D1 = X[:,0].reshape(X.shape[0],1) 
        D2 = X[:,0].reshape(X.shape[0],1)
        D4 = X[:,0].reshape(X.shape[0],1)
        
        # Real distance to the boundary
        # We need to do some rescaling to apply our already trained distance 
        X_new=self.rescale(X)
        D3 = self.Original_Distance.forward(X_new)
        
        return D1 , D2 ,D3 , D4 
        
    # Boundary function 
    def Boundary_extension(self,X):
        
        # Dirichlet condition on the entry, just constant values 
        G1 = self.RHO0*torch.ones(X.shape[0],1)
        G2 = self.VMOD0*torch.ones(X.shape[0],1)
        G4 = self.P0*torch.ones(X.shape[0],1) 
        
        # We can compute G3 analytically as a ruled surface 
        x=X[:,0] 
        y=X[:,1] 

        f1=self.f1(x,self.AL); 
        f2=self.f2(x,self.AL)
        
        # First we compute the angles
        f1x=torch.autograd.grad(f1,x,torch.ones_like(x),
                                retain_graph=True,create_graph=True)[0]
        f2x=torch.autograd.grad(f2,x,torch.ones_like(x),
                                retain_graph=True,create_graph=True)[0]
        
        theta1=torch.atan2(f1x,torch.ones_like(f1x))
        theta2=torch.atan2(f2x,torch.ones_like(f2x))
        
        # We just define G as a straigh line connecting theta1 and theta2
        G3=theta2 + (y-f2)*(theta1-theta2)/(f1-f2) 
        G3=G3.reshape(X.shape[0],1)
    
        return G1 , G2 , G3 , G4 
    
   # Function that computes the fluid variables and its derivatives
    def Fluid_Variables(self,X):
        
        # Compute the output of the model 
        y1,y2,y3,y4=self.forward(X)
        
        # We compute the boundary and distance extensions 
        G1 , G2 , G3 , G4 = self.Boundary_extension(X)
        D1 , D2 , D3 , D4 =self.Smooth_distance(X)
        
        # We compute flow variables 
        rho=G1+D1*y1
        vmod=G2+D2*y2
        theta=G3+D3*y3
        u=vmod*torch.cos(theta) # Horizontal component 
        v=vmod*torch.sin(theta) # Vertical component 
        p=G4+D4*y4
             
        return rho , u , v , p , vmod , theta 
    
    # Loss function, Conservation equations (Euler = inviscid)
    def loss_function(self,k=None,verbose=False):
        
        # We define the new domain 
        self.Domain.f1= lambda x : self.f1(x,self.AL)
        self.Domain.f2= lambda x : self.f2(x,self.AL)
        
        # We define the points where we are evaluating the loss
        X=self.Domain.Construct_points(self.batch_size) 
        x=X[:,0].reshape(self.batch_size,1); 
        x=torch.detach(x) ; x.requires_grad=True
        y=X[:,1].reshape(self.batch_size,1); 
        y=torch.detach(y) ; y.requires_grad=True 
        
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
        
        # Extra condition
        # We fix the mean value on the extrem of the horizontal velocity
        xp=torch.ones(self.batch_size,1,requires_grad=True)*self.Domain.Ix[1]
        
        # Equidistant grid 
        # yp=torch.linspace(float(self.f2(self.Domain.Ix[1],self.AL)),
        #                   float(self.f1(self.Domain.Ix[1],self.AL)),
        #                   self.batch_size,requires_grad=True).reshape(self.batch_size,1)

        # Monte Carlo grid
        I=self.f1(self.Domain.Ix[1],self.AL)-self.f2(self.Domain.Ix[1],self.AL)
        yp=self.f2(self.Domain.Ix[1],self.AL)+torch.rand([self.batch_size,1])*I
        yp=yp.reshape(self.batch_size).sort()[0].reshape(self.batch_size,1)
        X=torch.cat((xp,yp),dim=1)
        rho,u,v,p,_,_ = self.Fluid_Variables(X)
        UL=torch.trapz(u,yp,dim=0)/I
        
        L5 = abs(UL-self.UL)
        
        # Total form of the loss function 
        L=L1+L2+L3+L4+L5
        
        # Split of the loss function 
        if verbose: 
            print(f'Total Loss : {L}')
            print(f'Conservation of mass loss : {L1}')
            print(f'Conservation of momentum x loss : {L2}')
            print(f'Conservation of momentum y loss : {L3}')
            print(f'Conservation of energy loss : {L4}')
            print(f'Condition on x=L on horizontal velocity loss : {L5}')
            
        # We write the loss function in tensorboard
        if torch.utils.tensorboard.writer.SummaryWriter==type(self.writer) and type(k)==int:
            self.writer.add_scalar('L' , L , global_step=k)
            self.writer.add_scalar('L1', L1, global_step=k)
            self.writer.add_scalar('L2', L2, global_step=k)
            self.writer.add_scalar('L3', L3, global_step=k)
            self.writer.add_scalar('L4', L4, global_step=k)
            self.writer.add_scalar('L5', L5, global_step=k)
            
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
            if torch.utils.tensorboard.writer.SummaryWriter==type(self.writer):
                self.writer.add_scalar('AL',self.AL, global_step=k)
            # We print how the loss is evolving 
            if (k/(self.max_iter/5)==k//(self.max_iter/5)):
                print(f'\nPINN iteration : {k}. Loss function : {float(loss.item())}.'+
                      f' Area : {float(self.AL.data)}')            
    
    # Obtain the results after training 
    def Solution(self,X=None,N=2000):
        
        # If we do not choose especific points we pick random
        if X==None :
            # We update the domain (just in case) 
            self.Domain.f1= lambda x : self.f1(x,self.AL)
            self.Domain.f2= lambda x : self.f2(x,self.AL)
            # We construct the points that we need to evaluate the solution 
            X=self.Domain.Construct_points(N)
            
        if X[:,0].requires_grad==False:
            x=torch.detach(X[:,0]); x.requires_grad=True; x=x.reshape(X.shape[0],1)
            y=torch.detach(X[:,1]); y.requires_grad=True; y=y.reshape(X.shape[0],1)
            X=torch.cat((x,y),dim=1)
            
        # Computing the fluid variables in that points 
        rho,u,v,p,vmod,theta = self.Fluid_Variables(X) 
        
        return X , rho , u , v , p , vmod , theta 
    
    # Mass flux at a certain slice x 
    def One_Dimensional_results(self,x):
        # We update the domain (just in case) 
        self.Domain.f1= lambda x : self.f1(x,self.AL)
        self.Domain.f2= lambda x : self.f2(x,self.AL)
        
        xp=torch.ones(self.batch_size,1)*x
        yp=torch.linspace(float(self.Domain.f2(x)),
                          float(self.Domain.f1(x)),
                          self.batch_size).reshape(self.batch_size,1)
        X=torch.cat((xp,yp),dim=1)
        X , rho , u , _ , p , _ , _ = self.Solution(X)
        mass_flux=torch.trapz(rho*u,yp,dim=0)
        Ly=(float(self.Domain.f1(x))-
            float(self.Domain.f2(x)))
        rho1=torch.trapz(rho,yp,dim=0)/Ly
        u1=torch.trapz(u,yp,dim=0)/Ly
        p1=torch.trapz(p,yp,dim=0)/Ly 
        return  rho1, u1 , p1 , mass_flux 