import numpy as np

def value_iteration(env,theta=0.1,gamma=0.9):
    Q  = np.random.random(size=env.state_space_shape)
    Q[6,6]=np.zeros(shape=env.state_space_shape[2])
    while True:
        delta=0
        for i in range(Q.shape[0]):
            for j in range(Q.shape[1]):
                for k in range(Q.shape[2]):
                    q=Q[i,j,k]
                    max_reward=0
                    for l in range(len(env.action_space)):
                        reward=0
                        trans=env.trasitions(state=(i,j,k),action=l)
                        for x in trans:
                            reward += (x[2]*(x[1]+gamma*Q[x[0]]))
                        if reward>max_reward:
                            max_reward=reward
                    Q[i,j,k]=max_reward
                    delta=max(delta,abs(q-Q[i,j,k]))
        if delta<theta:
            break
        
    policy=np.empty_like(Q)
    for i in range(Q.shape[0]):
            for j in range(Q.shape[1]):
                for k in range(Q.shape[2]):
                    reward=[]
                    for l in range(len(env.action_space)):
                        temp=0
                        trans=env.trasitions(state=(i,j,k),action=l)
                        for x in trans:
                            temp += (x[2]*(m[1]+gamma*Q[x[0]]))
                        reward.append(temp)
                    policy[i,j,k]=np.argmax(reward)
    
    return policy
  
            
def run_policy(env,policy):
        state = env.state
        steps = 0
        while True:
            action=policy[state[0],state[1],state[2]]
            new_state,_,done=env.step(action)
            state=new_state
            steps+=1
            if steps==50 or done: break
            
        env.render()
        env.reset()