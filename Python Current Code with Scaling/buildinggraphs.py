import matplotlib.pyplot as plt
import numpy as np


X = np.pi
N = 8

x = np.linspace(0, X, 180)

selectivities_choice = [0, X/8, X/4, 3*X/8, X/2, 5*X/8, 3*X/4, 7*X/8]
selectivities = np.random.choice(selectivities_choice, N)

sigma = 0.5

for i in selectivities:
    y = np.exp(-1/2*(x-i)**2/(sigma**2))
    plt.plot(x,y, color='blue')
    tick_positions = np.linspace(0, X, 9)  # 0 to π in 8 equal steps
    tick_labels = [r'$0$'] + [rf'${i}\pi/8$' for i in range(1, 8)] + [r'$\pi$']


plt.xticks(tick_positions, tick_labels)
# plt.gca().get_yaxis().set_visible(False)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)

plt.show()