# Classes 
import matplotlib.pyplot as plt 
import numpy as np 
import torch 
from tqdm import tqdm 
# Domain that we are using to solve the Euler equations  
class Domain():
    def __init__(self,Ix,f1,f2): 

        # Geometry 
        self.Ix=torch.tensor([Ix[0],Ix[1]])
        self.f1 = f1 
        self.f2 = f2 
     
    # Method that randomly creates points inside the domain
    def Construct_points(self,batch_size): 
        x=self.Ix[0]+(self.Ix[1]-self.Ix[0])*torch.rand((batch_size,1))
        y=self.f2(x)+(self.f1(x)-self.f2(x))*torch.rand((batch_size,1))
        X=torch.cat((x,y),dim=1)
        return X 
        
    # Method that randomly creates points in the boundary 
    def Construct_boundary_points(self,b1=100,b2=400):
        
        # Points in DeltaOmega1
        x1=torch.ones(b1,1)*self.Ix[0]
        y1=self.f2(x1)+(self.f1(x1)-self.f2(x1))*torch.rand(b1,1)
        X1=torch.cat((x1,y1),dim=1)
        
        # Points in DeltaOmega2
        x2=self.Ix[0]+(self.Ix[1]-self.Ix[0])*torch.rand(b2,1)
        y2=self.f2(x2) + (self.f1(x2)-self.f2(x2))*torch.randint(0,2,size=(b2,1))
        X2=torch.cat((x2,y2),dim=1)
        
        # Border points 
        Xb=torch.cat((X1,X2),dim=0)
        
        return Xb 
    
    # Method that computes pairwise real distances between points in X and Xb 
    def Real_distance(self,Xb,X):
        distances=torch.zeros(X.shape[0],1)
        for index , point in enumerate(X): 
            local_distance=torch.ones_like(Xb)
            # Compute distance from the interior point to all boundary points 
            for index_b , point_b in enumerate(Xb):
                local_distance[index_b]=torch.norm(point_b-point)
            # Select the minimum distance 
            distances[index]=torch.min(local_distance)
        return distances 
    
    # Method that creates a heatmap between f1 and f2 
    def figure(self,variable_function,variable_name,figure_name,folder):
        
        # Meshgrid needed for the heatmap 
        X = np.linspace(self.Ix[0],self.Ix[1],1000)
        maxvalue = max(self.f1(torch.tensor(X)).detach().numpy())+0.2
        minvalue = min(self.f2(torch.tensor(X)).detach().numpy())-0.2
        Y = np.linspace(minvalue, maxvalue, 1000)
        X, Y = np.meshgrid(X, Y)
        X2 = X.ravel()
        Y2 = Y.ravel()
        X_NN=torch.cat((torch.tensor(X2).reshape(1000000,1),torch.tensor(Y2).reshape(1000000,1)),1)
        Y=variable_function(X_NN.float()).reshape(1000,1000)
        # Y=torch.flip(Y, dims=[0]).detach().numpy() # Somehow reshaping twice flip one direction. 
        Y=Y.detach().numpy() # There are sometimes that not flipped is needed. I think that it depends 
        # if the variable_function is a NN or a handmade function somehow. 
        
        # Define the region between the curves
        lower_curve = self.f2(torch.tensor(X)).detach().numpy()
        upper_curve = self.f1(torch.tensor(X)).detach().numpy() 
        
        # Create the plot
        fig, ax = plt.subplots()
        im = ax.imshow(Y,extent=[self.Ix[0],self.Ix[1],maxvalue,minvalue],
                       cmap='viridis',aspect='auto')
        ax.fill_between(X[0,:], upper_curve[0,:], 
                        np.ones_like(upper_curve[0,:])*maxvalue, color='white', alpha=1)
        ax.fill_between(X[0,:], lower_curve[0,:], 
                        np.ones_like(lower_curve[0,:])*minvalue, color='white', alpha=1)
        cbar = fig.colorbar(im)
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_xlim([self.Ix[0],self.Ix[1]])
        ax.set_ylim([minvalue,maxvalue])
        with torch.no_grad():
            x=torch.linspace(self.Ix[0],self.Ix[1],100)
            ax.plot(x,self.f1(x),linewidth=3,color='black',label=r'$\partial\Omega$')
            ax.plot(x,self.f2(x),linewidth=3,color='black')
            ax.plot(0*x+self.Ix[0],torch.linspace(float(self.f2(x[0])),
                                       float(self.f1(x[0])),100),color='black',linewidth=3)
        cbar.set_label(variable_name)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(folder+figure_name,dpi=100)

# NN that computes the distance smooth function
# We inherit the domain
class Distance(torch.nn.Module,Domain):
    def __init__(self,Ix,f1,f2):
        
        # Initialize both parents indepently 
        torch.nn.Module.__init__(self)
        Domain.__init__(self,Ix,f1,f2) 
        
        # NN parameter
        self.layer1=torch.nn.Linear(2,20)
        self.layer2=torch.nn.Linear(20,1)
        self.activation_function=torch.nn.Sigmoid()
        self.max_iter=5000
        
        # Interior points to impose real distance (FIXED during training)
        self.X3=self.Construct_points(100)
        self.Y3=self.Real_distance(self.Construct_boundary_points(),self.X3)
          
    def forward(self,x): 
        x = self.layer1(x)
        x = self.activation_function(x)
        x = self.layer2(x)
        x = self.activation_function(x)
        return x 
    
    def loss_function(self,verbose=False):
        
        # Boundary points 
        X=self.Construct_boundary_points()
        
        # We compute the estimation of the NN 
        YPRED=self.forward(X)
        YPRED3=self.forward(self.X3)
        
        if verbose: 
            print(sum(YPRED**2),sum((self.Y3-YPRED3)**2))
            
        return sum(YPRED**2) + 10*sum((self.Y3-YPRED3)**2)
    
    def fit(self):
        # We define the optimizer (Adam or SGD)
        optimizer=torch.optim.Adam(self.parameters(),lr=0.01)
        for k in tqdm(range(0,self.max_iter),desc='Distance loss'):
            # We reset the gradient of parameters
            optimizer.zero_grad()
            loss=self.loss_function()
            # Calculate backpropagation 
            loss.backward() 
            # We do one step of optimization 
            optimizer.step() 
            # writer.add_scalar('loss', loss.item(), global_step=k)
            # if (k/(self.max_iter/5)==k//(self.max_iter/5)):
                # print(f'Distance iteration : {k}. Loss function : {loss.item()}.')
        
# NN that computes the boundary function for the angle (G_phi)
# We inherit the domain 
class Boundary(torch.nn.Module,Domain):
    def __init__(self,THETA_inf,Ix,f1,f2):
        
        # Initialize both parents indepently 
        torch.nn.Module.__init__(self)
        Domain.__init__(self,Ix,f1,f2) 
        
        # Entry value 
        self.THETA_inf=THETA_inf 
        
        # NN parameters
        self.layer1=torch.nn.Linear(2,10)
        self.layer2=torch.nn.Linear(10,1)
        self.activation_function=torch.nn.Sigmoid()
        self.max_iter=5000
        
    def forward(self,x): 
        x = self.layer1(x)
        x = self.activation_function(x)
        x = self.layer2(x)
        return x 
    
    def loss_function(self):
        
        # We just need to force some conditions on the boundaries 
        
        # Points in DeltaOmega1
        x1=torch.ones(100,1)*self.Ix[0]
        y1=self.f2(x1)+(self.f1(x1)-self.f2(x1))*torch.rand(100,1)
        X1=torch.cat((x1,y1),dim=1)
        
        # Points in DeltaOmega2
        x2=self.Ix[0]+(self.Ix[1]-self.Ix[0])*torch.rand(400,1)
        x2.requires_grad=True 
        y2=self.f2(x2) + (self.f1(x2)-self.f2(x2))*torch.randint(0,2,size=(400,1))
        X2=torch.cat((x2,y2),dim=1)
        
        # We asign the labels for the points 
        
        # In DeltaOmega1 we set the constant dirichlet condition
        Y1=self.THETA_inf*torch.ones(size=(X1.shape[0],1))
        
        # In DeltaOmega2 we have to construct the tangent vector
        fx=torch.autograd.grad(y2,x2,torch.ones_like(y2),
                                retain_graph=True,create_graph=True)[0]
        
        Y2=torch.atan2(fx,torch.ones_like(fx))
        
        # We compute the estimation of the NN 
        YPRED1=self.forward(X1)
        YPRED2=self.forward(X2)
    
        return 4*sum((Y1-YPRED1)**2) + sum((Y2-YPRED2)**2) 
    
    def fit(self):
        # We define the optimizer (Adam or SGD)
        optimizer=torch.optim.Adam(self.parameters(),lr=0.01)
        for k in tqdm(range(0,self.max_iter),desc='Boundary loss'):
            # We reset the gradient of parameters
            optimizer.zero_grad()
            loss=self.loss_function()
            # Calculate backpropagation 
            loss.backward() 
            # We do one step of optimization 
            optimizer.step() 