import numpy as np
import imutils
import matplotlib.pyplot as plt
from scipy.integrate import quad


class WindField():    
    
    def __init__(self,size):
        
        self.size = size
        
        self.action_space = np.arange(8)
        self.state_space_shape=(size,size,8)
        
        self.state=(0,0,3)
        self.path=[(0,0)]
        
        self.targets=[]
        for i in range(2,5):
            self.targets.append((size-1,size-1)+(i,))  
            
        # self.wind_field=None
               
        
    def reset(self):
        
        self.state=(0,0,3)
        self.path=[(0,0)]
    

    def step(self,action):
        
        action_list = {
                            0:'N',
                            1:'NE',
                            2:'E',
                            3:'SE',
                            4:'S',
                            5:'SW',
                            6:'W',
                            7:'NW'
                        }
        
        state=self.state
        if isinstance(action,int):
            action=action_list[action]
        
        trans = self.trasitions(state,action)
        
        ran = np.random.random_sample()
        
        temp=0
        for i in trans:
            temp=temp+i[2]
            if ran<= temp:
                new_state=i[0]
                reward=i[1]
                break
        
        self.path.append(new_state[:2])
        
        done=False   
        if new_state in self.targets:
            new_state=(0,0,3)
            reward=0
            done=True
        
        self.state=new_state
        return [self.state,reward,done]
    
    
    def generate_windfield(self,type,varying=False):

            
            wind =[]
            for y in range(0,self.size):
                w =[]
                for x in range(0,self.size):
                    
                    if type==1 and not varying:
                        w.append(np.arctan(1))
                    elif type==1 and varying:
                        w.append(np.random.normal(np.arctan(1), np.pi/16))
                    
                    elif type==2 and not varying:
                        if x == float(self.size) - 1:
                            w.append(np.pi/2)
                        else:
                            w.append(np.arctan((y-self.size+1)/(x-self.size+1)))    
                    elif type==2 and varying:
                        if x == float(self.size) - 1:
                            w.append(np.random.normal(np.pi/2, np.pi/16))
                        else:
                            w.append(np.random.normal(np.arctan((y-self.size+1)/(x-self.size+1)), np.pi/16))
                    
                    elif type==3 and not varying:
                        if y==3:
                            w.append(0)
                        else:
                            w.append(np.arctan((x-3)/(y-3)))
                    elif type==3 and varying:
                        if y==3:
                            w.append(np.random.normal(0, np.pi/16))
                        else:
                            w.append(np.random.normal(np.arctan((x-3)/(y-3)), np.pi/16))
                            
                    elif type==4 and not varying:
                        if y==0:
                            w.append(0)
                        elif x==self.size-1:
                            w.append(np.pi/2)
                        else:
                            w.append(-np.arctan((x)/(y)))
                    elif type==4 and varying:
                        if y==0:
                            w.append(np.random.normal(0, np.pi/16))
                        elif x==self.size-1:
                            w.append(np.random.normal(np.pi/2, np.pi/16))
                        else:
                            w.append(np.random.normal(-np.arctan((x)/(y)), np.pi/16))
                wind.append(w)

            wind_field = []
            for y in range(0,self.size):
                W_x = []
                for x in range(0,self.size):
                    if varying:
                        u = np.cos(wind[y][x])*np.random.normal(10, 5)
                        v = np.sin(wind[y][x])*np.random.normal(10, 5)
                    else:
                        u = np.cos(wind[y][x])
                        v = np.sin(wind[y][x])
                    W_x.append([u,v])
                wind_field.append(W_x)

            wind_field = np.array(wind_field)

            if varying:
                self.wind_field=wind_field
            
            return wind_field
        
        
    def reward(self,state,c=30,w_max=15):

            wind_field=self.wind_field
            
            [u,v]=wind_field[state[0],state[1]]
            if u !=0 and self.size-1-state[0] != 0:
                reward=c*(np.sqrt(u**2+v**2)*np.cos(np.arctan(v/u)+np.arctan((self.size-1-state[1])/(self.size-1-state[0]))))/w_max
            elif state[0]==6:
                reward=c*(np.sqrt(u**2+v**2)*np.cos(np.arctan(v/u)+np.pi/2))/w_max
            elif u==0:
                reward=c*(np.sqrt(u**2+v**2)*np.cos(np.pi/2+np.arctan((self.size-1-state[1])/(self.size-1-state[0]))))/w_max
            else:
                reward=-1*c*(np.sqrt(u**2+v**2))/w_max
                
            if tuple(state) in self.targets:
                reward=2*c
            else:
                reward-=5
                
            return reward
    
    
    def trasitions(self,state,action,sigma=np.pi/12,v_min=20):
        
        def integrand(x):
            return (np.e**((-1/2)*((x-omega)/sigma)**2))/(sigma*np.sqrt(2*np.pi))
        
        wind_field=self.wind_field
        
        angle_lookup = {
                        'N': -np.pi/2,
                        'NE': -np.pi/4,
                        'E': 0,
                        'SE': np.pi/4,
                        'S': np.pi/2,
                        'SW': 3*np.pi/4,
                        'W': np.pi,
                        'NW': 5*np.pi/4
                       }
        
        action_lookup = {
                            'N':0,
                            'NE':1,
                            'E':2,
                            'SE':3,
                            'S':4,
                            'SW':5,
                            'W':6,
                            'NW':7
                        }
        
        inv_action_lookup = {
                            0:'N',
                            1:'NE',
                            2:'E',
                            3:'SE',
                            4:'S',
                            5:'SW',
                            6:'W',
                            7:'NW'
                        }
        
        change_coordinates = {
                                'N':np.array([0,1,0]),
                                'NE':np.array([1,1,0]),
                                'E':np.array([1,0,0]),
                                'SE':np.array([1,-1,0]),
                                'S':np.array([0,-1,0]),
                                'SW':np.array([-1,-1,0]),
                                'W':np.array([-1,0,0]),
                                'NW':np.array([-1,1,0])
                                }
        
        action=inv_action_lookup[action]
        
        x = state[0]
        y = state[1]
        psi = angle_lookup[action]
        
        F_x = wind_field[y,x,0] + v_min*np.cos(psi)
        F_y = wind_field[y,x,1] + v_min*np.sin(psi)
        
        omega = np.arctan(F_y/F_x)
        F_mag = np.sqrt(F_x**2 + F_y**2)
        
        # integration over gaussian for all actions
        P = []
        p_total=0
        for i in range(8):
            I = quad(integrand, float(list(angle_lookup.values())[i]) - np.pi/8, float(list(angle_lookup.values())[i]) + np.pi/8)
            next_state=np.concatenate(((np.array(state)+np.array(list(change_coordinates.values())[i]))[:2],np.array([action_lookup[action]])))
            if next_state[0]>=0 and next_state[0]<self.size and next_state[1]>=0 and next_state[1]<self.size:
                transition_reward=self.reward(next_state)
                p_total+=I[0]
                P.append([tuple(next_state),transition_reward,I[0]])
        P=np.array(P,dtype=object)
        P[:,2]=P[:,2]/p_total
        
        return P
    
    
    def plot_windfield(self,wind_field):

        x,y = np.meshgrid(np.linspace(0,6,7), np.linspace(0,6,7))
        u = wind_field[:,:,0]
        v = wind_field[:,:,1]
        plt.quiver(x,-y,u,-v)
        
        
    def render(self):
        canvas=np.zeros((100*self.size,100*self.size,3),dtype=np.uint8)
        
        triangle=imutils.resize(255*plt.imread(r"triangle.png")[:,:,:3],width=50)
        canvas[25:75,25:75]=triangle
        
        circle=imutils.resize(255*plt.imread(r"circle.png")[:,:,:3],width=50)
        canvas[(100*self.size)-75:(100*self.size)-25,(100*self.size)-75:(100*self.size)-25]=circle
        
        airplane=imutils.resize(255*plt.imread(r"images.png")[:,:,:3],width=100)
        pos=(100*self.path[-1][1],100*self.path[-1][0])
        facing=self.state[2]
        facing_img={
                    0:airplane,
                    1:imutils.rotate(airplane,angle=360-45),
                    2:imutils.rotate(airplane,angle=360-90),
                    3:imutils.rotate(airplane,angle=360-135),
                    4:imutils.rotate(airplane,angle=360-180),
                    5:imutils.rotate(airplane,angle=360-225),
                    6:imutils.rotate(airplane,angle=360-270),
                    7:imutils.rotate(airplane,angle=360-315)
                    }
        canvas[pos[0]:100+pos[0],pos[1]:100+pos[1]]=facing_img[facing]
        plt.imshow(canvas)
        
        if len(self.path)>1:
            for i in range(len(self.path)-1):
                plt.plot([100*self.path[i][0]+50,100*self.path[i+1][0]+50],[100*self.path[i][1]+50,100*self.path[i+1][1]+50],color="white", linewidth=1)
        
        plt.grid(color="white")
        plt.xticks(np.arange(0,100*self.size+1,100),(np.arange(0,100*self.size+1,100)/100).astype(np.int0))
        plt.yticks(np.arange(0,100*self.size+1,100),(np.arange(0,100*self.size+1,100)/100).astype(np.int0))
        plt.tick_params(axis='y', labelright=True,labelleft=False)
        plt.show()
        