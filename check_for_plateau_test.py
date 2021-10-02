# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
def has_plateaued(reward_history, patience=500):
    ''' Checks if the avg rewards are plateauing. If the mean of last 100 and 300 are different within 5%, it's plauteauing. '''

    single_patience_mean = np.mean(reward_history[-patience:])
    double_patience_mean = np.mean(reward_history[-2*patience:])

    if len(reward_history) < 2*patience:
        return False

    plateauing_bool = np.abs((single_patience_mean - double_patience_mean) / single_patience_mean)*100 < 0.1

    return plateauing_bool

class OUActionNoise:
    ''' Orenstein-Uhlenbeck Noise implementation. Temporally correlated noise with mu=0.'''
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()
    
    def __call__(self):
        ''' Enables the operation below:
            noise = OUActionNoise()
            noise() '''
        x = self.x_prev + self.theta * (self.mu-self.x_prev)*self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

# %%
noise = OUActionNoise(mu=np.array([10]))

val = []
while True:
    value = noise()
    val.append(value)
    if has_plateaued(val):
        break

plt.plot(val)

# %%
plt.plot(val)
# %%
for i in range(1000):
    temp = val[:i]
    if has_plateaued(temp):
        print(i)
        break


# %%
type(noise())
# %%

# %%
