import numpy as np

print('Kepler 1520b')
P = .6535538
R_s = 4.522e8
a = 1.9448e9
R_p = 5e5
k = R_p/R_s
b = [0, 0.5, 1]
for b_0 in b:
    i = np.arccos((b_0 * R_s)/a)
    T = (1/np.pi) * np.arcsin((R_s/a) * ((((1 + k)**2 - b_0**2)**0.5)/np.sin(i)))
    print(b_0, T)

print('K2-22b')
P = .381078
R_s = 3.965e8
a = 1.316e9
for b_0 in b:
    i = np.arccos((b_0 * R_s)/a)
    T = (1/np.pi) * np.arcsin((R_s/a) * ((((1 + k)**2 - b_0**2)**0.5)/np.sin(i)))
    print(b_0, T)
