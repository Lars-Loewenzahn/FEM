# Programmieraufgabe 1 a)
L = [[1,0,0],[-2,1,0],[4,5,1]]
R = [[2,-1,6],[0,3,9],[0,0,-2]]
b = [18,-3, 231]

def vorrueck(L,R,b):
    n = len(L)
    y = [0]*n
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i][j]*y[j]
        y[i] /= L[i][i]

    x = [0]*n
    for i in range(n-1, -1, -1):
        x[i] = y[i]
        for j in range(i+1,n):
            x[i] -= R[i][j]*x[j]
        x[i] /= R[i][i]
    return x

x = vorrueck(L,R,b)
print("1a:")
print(x)

# Programmieraufgabe 1 b)
def lr(matrix):
    n = len(matrix)
    L = create_identity_matrix(n)
    R = matrix
    for i in range(0,n-1): # i Spalte
        for j in range(i+1, n):  # j Zeile
            div = R[j][i]/R[i][i]
            R[j] = vector_add(R[j],vector_multiply(R[i], -div))
            L[j][i] = div
    return L,R

def multiply(A,B):
    C = []
    try:
        b_rows = len(B[0])
    except:
        b_rows = 1

    for a_line in A:
        c_line = []
        for j in range(b_rows):
            c_i = 0
            for k, a in enumerate(a_line):
                c_i += a*B[k][j]
            c_line.append(c_i)
        C.append(c_line)
    return C

L = [[1,0,0],[-2,1,0],[4,5,1]]
R = [[2,-1,6],[0,3,9],[0,0,-2]]
C = multiply(L,R)
print("1b:")
print(C)
L,R = lr(C)
print(L)
print(R)
C = multiply(L,R)
print(C)


# Programmieraufgabe 1 c)
print()
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top']       = True
mpl.rcParams['ytick.right']     = True

# --- erstes Teilfenster: y = sin(5x) und y = cos(5x) --
plt.figure(figsize=(7,5.7))
plt.subplot(2, 1, 1)              # 2 Zeilen, 1 Spalte, Plot 1
x1 = np.linspace(0, 1, 500)
plt.plot(x1, np.sin(5*x1), 'black', lw=0.7)        # sin(5x)
plt.plot(x1, np.cos(5*x1), 'black', lw=0.7)        # cos(5x)
plt.title(r'y=sin(5x) und y=cos(5x)', fontsize=8)
plt.xlabel('x')
plt.xlim(0, 1)
labels = (0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)
plt.xticks(tuple(x*0.1 for x in range(0,11)), labels)    # 0,0.1,0.2,…,1.0
plt.ylabel('y')
plt.yticks(tuple(x for x in range(-2,4)))       # -2,-1,0,1,2
plt.grid(True, linestyle=':')        # gestricheltes Gitter
plt.ylim(-2,2)


# --- zweites Teilfenster: y = sin(exp(x)) ---
plt.subplot(2, 1, 2)              # 2 Zeilen, 1 Spalte, Plot 2
x2 = np.linspace(-3, 3, 1000)
plt.plot(x2, np.sin(np.exp(x2)), 'black', lw=0.7)
plt.title(r'y=sin(exp(x))', fontsize=8)
plt.xlabel('x')
plt.xlim(-3,3)
plt.xticks(tuple(x for x in range(-3,4)))       
plt.ylabel('y')
plt.ylim(-1.5, 1.5)                   
labels = (-1.5,-1,-0.5,0,0.5,1,1.5)
plt.yticks(tuple(x*0.1 for x in range(-15,16,5)), labels)   # 
plt.grid(False)

plt.tight_layout()  # Zwischenräume anpassen
plt.show()