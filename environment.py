import numpy as np
from math import pi,e


class WindField():
    
    def __init__(self):
        
        self.size = 7
        self.action_space = np.arange(8)
        self.state_space_shape=(self.size,self.size,8)
        self.state=(0,0,3)
        self.path=[(0,0)]
        self.targets=[(self.size-1,self.size-1)+(i,) for i in range(2,5)]

              
    def reset(self):
        
        self.state=(0,0,3)
        self.path=[(0,0)]

    
    def uniform_windfield(self, stochastic = True, random_mag=True):
        x0 = 7
        y0 = 7

        X = np.arange(0,7,1)
        Y = np.arange(0,7,1)
        
        if stochastic:
            sigma = pi/16
        else: sigma = 0

        w_xy =[]
        for y in Y:
            w_x =[]
            for x in X:
                w_x.append(np.random.normal(np.arctan(((y-y0)/(x-x0))), sigma))
            w_xy.append(w_x)

        if random_mag:
            mag = np.random.normal(10,5)
        else: mag = 10 
        W = []
        for y in Y:
            W_x = []
            for x in X:
                u = np.cos(w_xy[y][x])*mag
                v = np.sin(w_xy[y][x])*mag
                W_x.append([u,v])
            W.append(W_x)
        W = np.array(W)
        self.wind = W
        return W

    
    def non_uniform_windfield1(self, stochastic = True, random_mag = True):
        x0 = 7
        y0 = 7

        X = np.arange(0,7,1)
        Y = np.arange(0,7,1)

        if stochastic:
            sigma = pi/16
        else: sigma = 0

        w_xy =[]
        for y in Y:
            w_x =[]
            for x in X:
                if y == 0:
                    w_x.append(np.random.normal(0, sigma))
                elif x==6:
                    w_x.append(np.random.normal(pi/2, sigma))
                else:
                    w_x.append(np.random.normal(-np.arctan((x/y)), sigma))
            w_xy.append(w_x)

        if random_mag:
            mag = np.random.normal(10,5)
        else: mag = 10 

        W = []
        for y in Y:
            W_x = []
            for x in X:
                u = np.cos(w_xy[y][x])*mag
                v = np.sin(w_xy[y][x])*mag
                W_x.append([u,v])
            W.append(W_x)
        W = np.array(W)
        self.wind = W
        return W

    
    def non_uniform_windfield2(self):
        x0 = 7
        y0 = 7

        X = np.arange(0,7,1)
        Y = np.arange(0,7,1)

        w_xy =[]
        for y in Y:
            w_x =[]
            for x in X:
                if x == 3 and y == 3:
                    w_x.append(np.random.normal(pi/2,pi/16))
                elif y>3:
                    w_x.append(np.random.normal(np.arctan(((-3+x)/(3-y))), pi/16))
                else: w_x.append(np.random.normal(np.arctan(((-3+x)/(3-y)))-pi, pi/16))
            w_xy.append(w_x)
            
        W = []
        for y in Y:
            W_x = []
            for x in X:
                u = np.cos(w_xy[y][x])*10
                v = np.sin(w_xy[y][x])*10
                W_x.append([u,v])
            W.append(W_x)
        W = np.array(W)
        self.wind = W
        return W
    

    def plot_windfield(self,wind):
        import matplotlib.pyplot as plt

        x,y = np.meshgrid(np.linspace(0,6,7), np.linspace(0,6,7))
        u = wind[:,:,0]
        v = wind[:,:,1]
        plt.quiver(x,-y,u,-v)
    

    def reward(self,state,c=30):

            w_max = 15
            wind=self.wind
            
            x = state[0]
            y = state[1]
            u = wind[x,y][0]
            v = wind[x,y][1]
            if u !=0 and 6-state[0] != 0: reward=c*(np.sqrt(u**2+v**2)*np.cos(np.arctan(v/u)+np.arctan((6-state[1])/(6-state[0]))))/w_max
            elif state[0]==6: reward=c*(np.sqrt(u**2+v**2)*np.cos(np.arctan(v/u)+pi/2))/w_max
            elif u==0: reward=c*(np.sqrt(u**2+v**2)*np.cos(pi/2+np.arctan((6-state[1])/(6-state[0]))))/w_max
            else: reward=-1*c*(np.sqrt(u**2+v**2))/w_max
            state = tuple(state)

            if state in self.targets: reward=3*c
            else: reward-=c/5
                
            return reward


    def trasitions(self,state,action,sigma=pi/16):
        from scipy.integrate import quad
        
        V = 20
        wind=self.wind

        action_list = {0:'N', 1:'NE', 2:'E', 3:'SE', 4:'S', 5:'SW', 6:'W', 7:'NW'}
        
        action_angle = {0: -pi/2, 1: -pi/4, 2: 0, 3: pi/4, 4: pi/2, 5: 3*pi/4, 6: pi, 7: 5*pi/4}
        
        move_states = {0:np.array([0,1,0]), 1:np.array([1,1,0]), 2:np.array([1,0,0]), 3:np.array([1,-1,0]), 4:np.array([0,-1,0]), 5:np.array([-1,-1,0]), 6:np.array([-1,0,0]), 7:np.array([-1,1,0])}
        
        x = state[0]
        y = state[1]
        
        F_x = wind[y,x,0] + V*np.cos(action)
        F_y = wind[y,x,1] + V*np.sin(action)
        
        w = np.arctan(F_y/F_x)
        F_mag = np.sqrt(F_x**2 + F_y**2)

        Theta = list(action_angle.values())[i]

        def integrand(x):
            return (e**((-1/2)*((x-w)/sigma)**2))/(sigma*np.sqrt(2*pi))
        trans = []
        p_sum = 0
        for i in range(8):
            I = quad(integrand, float(Theta[i]) - pi/8, float(Theta[i]) + pi/8)
            next_state = np.array(state)+np.array(list(move_states.values())[i])
            next_state = np.append(next_state[:2],action)
            if next_state[0]>=0 and next_state[0]<7 and next_state[1]>=0 and next_state[1]<7:
                transition_reward=self.reward(next_state)
                p_sum+=I[0]
                trans.append([tuple(next_state),transition_reward,I[0]])
        trans=np.array(trans,dtype=object)
        trans[:,2]=trans[:,2]/p_sum
        
        return trans


    def step(self,action):
        
        action_list = {0:'N', 1:'NE', 2:'E', 3:'SE', 4:'S', 5:'SW', 6:'W', 7:'NW'}
        state=self.state

        trans = self.trasitions(state,action)
        n = np.random.random_sample()
        p_sum=0
        for i in trans:
            p_sum = p_sum + i[2]
            if n <= temp:
                new_state = i[0]
                reward = i[1]
                break
        
        self.path.append(new_state[:2])
        
        done=False   
        if new_state in self.targets:
            new_state=(0,0,3)
            reward=0
            done=True
        
        self.state=new_state
        return [self.state,reward,done]
        
        
    def render(self):
        import imutils
        import matplotlib.pyplot as plt

        background = np.zeros((100*self.size,100*self.size,3),dtype=np.uint8)
        Start = imutils.resize(255*plt.imread(r"start.png")[:,:,:3],width=50)
        background[25:75,25:75] = Start
        End=imutils.resize(255*plt.imread(r"end.png")[:,:,:3],width=50)
        background[(100*self.size)-75:(100*self.size)-25,(100*self.size)-75:(100*self.size)-25] = End
        plane = imutils.resize(255*plt.imread(r"plane.png")[:,:,:3],width=100)
        pos=(100*self.path[-1][1],100*self.path[-1][0])
        facing=self.state[2]
        facing_img={0:plane, 1:imutils.rotate(plane,angle=360-45), 2:imutils.rotate(plane,angle=360-90), 3:imutils.rotate(plane,angle=360-135), 4:imutils.rotate(plane,angle=360-180), 5:imutils.rotate(plane,angle=360-225), 6:imutils.rotate(plane,angle=360-270), 7:imutils.rotate(plane,angle=360-315)}
        background[pos[0]:100+pos[0],pos[1]:100+pos[1]]=facing_img[facing]
        plt.imshow(background)
        
        if len(self.path)>1:
            for i in range(len(self.path)-1):
                plt.plot([100*self.path[i][0]+50,100*self.path[i+1][0]+50],[100*self.path[i][1]+50,100*self.path[i+1][1]+50],color="white", linewidth=1)
        
        plt.grid(color="white")
        plt.xticks(np.arange(0,100*self.size+1,100),(np.arange(0,100*self.size+1,100)/100).astype(np.int0))
        plt.yticks(np.arange(0,100*self.size+1,100),(np.arange(0,100*self.size+1,100)/100).astype(np.int0))
        plt.tick_params(axis='y', labelright=True,labelleft=False)
        plt.show()
        