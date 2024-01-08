import random
import numpy as np
import matplotlib.pyplot as plt

def c3_spin_config_init(size, spin, theta):
    # theta is the angle between spin and field, 0<=theta<=pi/3 (radian)
    s = np.zeros((size+2, size+2))
    for a in range(1, size+1):
        for b in range(1, size+1):
            rand_num = 2 * random.randint(0, 1) - 1  # +1 or -1
            if rand_num > 0:
                s[a+1, b+1] = rand_num * spin * np.cos(theta)
            else:
                s[a+1, b+1] = rand_num * spin * np.cos(np.pi/3 - theta)
    # periodic boundary condition
    s[:, 0] = s[:, size]
    s[:, size+1] = s[:, 1]
    s[0, :] = s[size, :]
    s[size+1, :] = s[1, :]
    # output +/- spin
    spin_plus = spin * np.cos(theta)
    spin_minus = -spin * np.cos(np.pi/3 - theta)
    return s, spin_plus, spin_minus

def mc_flip(spin_config, spin_plus, spin_minus, i, j, T, J, E, size):
    # T = temperature, J = nearest-neighbor exchange, E = external field
    k = 8.6173303e-5  # Boltzmann Constant
    s = spin_config
    if s[i, j] > 0:
        E_diff = J * (s[i, j] - spin_minus) * (s[i-1, j] + s[i+1, j] + s[i, j-1] + s[i, j+1] + s[i+1, j-1] + s[i-1, j+1]) + E * (s[i, j] - spin_minus)
    else:
        E_diff = J * (s[i, j] - spin_plus) * (s[i-1, j] + s[i+1, j] + s[i, j-1] + s[i, j+1] + s[i+1, j-1] + s[i-1, j+1]) + E * (s[i, j] - spin_plus)
    
    if E_diff <= 0 or random.random() < np.exp(-E_diff / (T * k)):
        if s[i, j] > 0:
            s[i, j] = spin_minus
        else:
            s[i, j] = spin_plus

    # update periodic boundary
    s[:, 0] = s[:, size]
    s[:, size+1] = s[:, 1]
    s[0, :] = s[size, :]
    s[size+1, :] = s[1, :]
    return s

size = 30
spin = 1
J = 3
T = 2
data = []
# s_tmp, spin_plus, spin_minus = c3_spin_config_init(size, spin, np.pi/3)
s_tmp, spin_plus, spin_minus = c3_spin_config_init(size, spin, np.pi*4/18)
E_list = [0] * 10 + list(range(0, 31, 2)) + (list(range(30, -31, -2)) + list(range(-30, 31, 2))) * 10

# Main starts
# for E in range(-100, 101, 5):
for E in E_list:
    mcs = 0
    ct = 0
    S = 0
    while mcs < 10000:
        i = random.randint(1, size)  # pick site randomly
        j = random.randint(1, size)  # pick site randomly
        s_tmp = mc_flip(s_tmp, spin_plus, spin_minus, i, j, T, J, E, size)
        mcs += 1
        if mcs > 8000:
            S += np.sum(s_tmp[1:size+1, 1:size+1]) / size**2
            ct += 1

    spin_site = S / ct  # spin per site
    # print(spin_site)
    data.append(np.array([E, spin_site]))
data = np.array(data)
np.savetxt('data_theta_60_test.txt', data)

plt.figure(figsize=(4,4))
plt.axvline(x=0, linestyle='--', color='k', linewidth=1)
plt.axhline(y=0, linestyle='--', color='k', linewidth=1)
# plt.plot(data[:, 0], data[:, 1:], '-o', label=[r'$\theta$=0', r'$\theta$=$\pi$/6', r'$\theta$=$\pi$/3'])
plt.plot(data[:, 0], data[:, 1], '-ro')
# plt.legend(loc='lower right', fontsize=10)
plt.tick_params(axis='y', direction='in')
plt.xticks([])
plt.gca().set_xticks([])
# plt.xlabel('Eletric field (a.u.)', fontsize=15)
# plt.ylabel('Polarization (a.u.)', fontsize=15)
# plt.show()

plt.savefig('hysteresis_60.png', dpi=800)