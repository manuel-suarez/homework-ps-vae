# Setup
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import StrMethodFormatter
import numpy as np
from numpy.random import random, randint, randn
from zernike import RZern

# Generación de los polinomios
# Definimos parámetros iniciales
order = 6           # Orden de los polinomios a generar
cart = RZern(order) # Generador de polinomios
dim = 128           # Tamaño de dimensiones de imagen
L, K = dim, dim     # Tamaño de cada imagen
num = 30000         # Tamaño conjunto de entrenamiento
num_test = 5000     # Tamaño conjunto de prueba
num_epochs = 50     # Número de épocas para el entrenamiento
latent_dim = 10      # Dimensión del espacio latente de la red VAE
# Definimos el grid para la generación de las señales
ddx = np.linspace(-1.0, 1.0, K)
ddy = np.linspace(-1.0, 1.0, L)
xv, yv = np.meshgrid(ddx, ddy)
cart.make_cart_grid(xv, yv)
# Generamos 30,000 señales variando los parámetros del polinomio a generar
Z = []
for i in range(num):
  # Variamos los parámetros para el polinomio
  c = np.random.normal(size=cart.nk)
  # Generamos polinomio
  Phi = cart.eval_grid(c, matrix=True)
  # Reemplazamos NaN con 0
  p = np.nan_to_num(Phi, False, 0)
  #p = Phi
  # Reescalamos a 0-1 (necesario para que la red calcule correctamente las métricas)
  # Verificar con el Dr. si esto sería necesario en este caso
  p_scaled = (p - np.min(p)) / (np.max(p) - np.min(p))
  # Agregamos a conjunto de resultados
  Z.append(p_scaled)

# Desplegamos una muestra
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
for i in range(3):
  for j in range(3):
    # print(Z[i*3 + j])
    ax[i,j].imshow(Z[i*3 + j])
    ax[i,j].set_title(f"Señal {i*3+j}")

fig.tight_layout()
plt.savefig('figure_1.png')

fig, ax = plt.subplots(nrows=3, ncols=3, subplot_kw={"projection": "3d"})
for i in range(3):
  for j in range(3):
    ax[i, j].plot_surface(ddx, ddy, Z[i * 3 + j], cmap=cm.coolwarm,
                          linewidth=0, antialiased=False)
    ax[i, j].set_title(f"Señal {i * 3 + j}")

fig.tight_layout()
plt.savefig('figure_2.png')

# Generamos las derivadas direccionales para cada imagen
Dx = []
Dy = []
for img in Z:
  img_dy, img_dx = np.gradient(img)
  Dx.append(img_dx)
  Dy.append(img_dy)

# Desplegamos una muestra de los gradientes en X
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
for i in range(3):
  for j in range(3):
    ax[i,j].imshow(Dx[i*3 + j])
    ax[i,j].set_title(f"Dx {i*3+j}")

fig.tight_layout()
plt.savefig('figure_3.png')