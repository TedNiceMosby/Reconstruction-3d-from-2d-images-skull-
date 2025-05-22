from skimage import io, color, filters, morphology, measure
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import cv2 
import os
plt.close('all')


folder_path = "Este es la carpeta donde se encuentran todas las imágenes"
file_paths = []

for filename in os.listdir(folder_path):
    full_path = os.path.join(folder_path, filename)
    file_paths.append(full_path)


#%% PREPROCESAMIENTO

def leerImagen(ruta):
    imagen = io.imread(ruta)
    if imagen.ndim == 3:
        imagen_gris = color.rgb2gray(imagen)
    else:
        imagen_gris= imagen
        
    return imagen_gris

def preprocesamiento(imagen_gris):
    imagen_gris = np.uint8(imagen_gris*255)
    #imagen_gris_cortada = imagen_gris[0:710, 230:800]
    imagen_gris_cortada = imagen_gris[0:710, 230:760]
    gauss = cv2.medianBlur(imagen_gris_cortada, 5)    
    return gauss

def binarizarImagen(imagenPre):
    umbral = threshold_otsu(imagenPre)
    binarizada = imagenPre > umbral  #Tiene valores booleanos 
    binarizada_limpia = morphology.remove_small_objects(~binarizada, min_size=40)# Elimina las islas blancas originalmente
                                                                        # se debe de invertir
    return binarizada_limpia
                                                                        
def coordenadasContorno(binarizada):
    contornos = measure.find_contours(binarizada, level=0.5)

    #     El modificar el valor de level implica que:
    #         level=0.8: Detectará contornos más dentro del objeto
    #         level=0.1: Detectará contornos más fuera del objeto

    if len(contornos) == 0:
        return [], []
    contorno_principal = max(contornos, key=len)
    x = contorno_principal[:, 1]
    y = contorno_principal[:, 0]
    return x, y

#%% OBTENIENDO COORDENADAS DEL CONTORNO

X = []
Y = []
cont = 0
for i in file_paths:
    # if cont == 25:
    #     break
    imagen = leerImagen(i)
    imagenPre = preprocesamiento(imagen)
    binaria = binarizarImagen(imagenPre)
    x, y =coordenadasContorno(binaria)
    x = x - np.mean(x)  # centro horizontal (radio)
    y = y - np.mean(y)  # centro vertical (altura)
    X.append(x)
    Y.append(y)
    cont +=1

#6, 15, 26, 37, 48, 59, 70, 81, 92
#X=X[0:10]
#Y=Y[0:10]

# LIMPIEZA DE PERFILES
indices_a_eliminar = [5, 15, 26, 37, 48, 59, 70, 81, 92]
X = [x for i, x in enumerate(X) if i not in indices_a_eliminar]
Y = [y for i, y in enumerate(Y) if i not in indices_a_eliminar]


#%% INTERPOLACION DE CAPAS
num_intermedios = 3  # Num de capas intermedias entre cada par

X_interp = []
Y_interp = []

for i in range(len(X) - 1):
    x1, y1 = X[i], Y[i]          # perfil original i
    x2, y2 = X[i+1], Y[i+1]      # perfil original i+1

    tree = cKDTree(np.column_stack([x2, y2]))  # construye árbol del segundo perfil

    for n in range(1, num_intermedios + 1):
        alpha = n / (num_intermedios + 1)      # peso de interpolación
        x_gen = []
        y_gen = []

        for xa, ya in zip(x1, y1):
            _, idx = tree.query([xa, ya])      # busca punto más cercano en el perfil i+1
            xb, yb = x2[idx], y2[idx]          # punto correspondiente

            # interpolación lineal entre el punto de x1 y su par más cercano en x2
            xi = (1 - alpha) * xa + alpha * xb
            yi = (1 - alpha) * ya + alpha * yb

            x_gen.append(xi)
            y_gen.append(yi)

        # guarda una nueva capa intermedia
        X_interp.append(np.array(x_gen))
        Y_interp.append(np.array(y_gen))


X_total = []
Y_total = []

for i in range(len(X) - 1):
    X_total.append(X[i])# perfil original
    Y_total.append(Y[i])

    for j in range(num_intermedios):
        index = i * num_intermedios + j  # esto asegura que tomamos los bloques correctos
        X_total.append(X_interp[index])  # capa interpolada 1
        Y_total.append(Y_interp[index])  # capa interpolada 1

# ***no olvides el último perfil
X_total.append(X[-1])
Y_total.append(Y[-1])



print("Total perfiles reales:", len(X))
print("Total interpolados:", len(X_interp))
print("Total final:", len(X_total))  # debería ser: (N-1)×(intermedios+1) + 1



def filtrar_extremos_de_contorno(xi, yi, extremos=250, umbral_radio=130, delta_y=3):
    
    x_filtrado = []
    y_filtrado = []
    n = len(xi)
    #Filtramos solo los ultinmos puntos de un perfil si son planos o tienen radio muy alto.
    #Conserva todo lo demás sin filtrar.

    for i in range(n):
        # Solo filtrar si es un punto en el extremo final del perfil
        if i >= n - extremos: # etsa es la condicion
            y_dif1 = abs(yi[i] - yi[i - 1]) if i > 0 else 0
            y_dif2 = abs(yi[i] - yi[i + 1]) if i < n - 1 else 0
            radio = np.sqrt(xi[i] ** 2 + yi[i] ** 2)

            # Elimina solo si el punto es sospechoso
            if radio > umbral_radio and y_dif1 < delta_y and y_dif2 < delta_y:
                continue  # descartar el punto

        x_filtrado.append(xi[i])
        y_filtrado.append(yi[i])

    return np.array(x_filtrado), np.array(y_filtrado)



#%%  TRANSFORMACION DE COORDENADAS

X_rot = []
Y_rot = []
Z_rot = []

n = len(X)
n_total = len(X_total)
theta = np.linspace(0, 2* np.pi, n_total)  


for i, (xi, yi) in enumerate(zip(X_total, Y_total)):  # Aquí, xi es la altura (eje Z), yi es radio
    angulo = theta[i]
    #Filtramos el ruido
    xi, yi = filtrar_extremos_de_contorno(xi, yi)
    for x_punto, y_punto in zip(xi, yi):
        
        x_rot = y_punto * np.cos(angulo)  # eje X
        y_rot = y_punto * np.sin(angulo)  # eje Y
        z_rot = x_punto                   # eje Z (altura del perfil)

        X_rot.append(x_rot)
        Y_rot.append(y_rot)
        Z_rot.append(z_rot)



#%% MOSTRAR LA FIGURA A TRAVES DE CAPAS

#Las capas nos sirve para ver si estan alineados

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Convertir a numpy arrays
Xr = np.array(X_rot)
Yr = np.array(Y_rot)
Zr = np.array(Z_rot)

#Filtro extra
z_min_umbral = 201
mascara = Zr < z_min_umbral

Xr = Xr[mascara]
Yr = Yr[mascara]
Zr = Zr[mascara]

num_capas = 10
z_min, z_max = np.min(Zr), np.max(Zr)
capas = np.linspace(z_min, z_max, num_capas + 1)

# Dibujar cada capa con un color diferente
for i in range(num_capas):
    z_low = capas[i]
    z_high = capas[i + 1]
    idx = (Zr >= z_low) & (Zr < z_high)

    ax.scatter(Xr[idx], Yr[idx], Zr[idx], s=0.5)

ax.invert_zaxis() #Invertimos el eje z para que se muestre de forma correca
ax.set_title("Cráneo en 3D", fontdict={'fontstyle': 'italic'})
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.view_init(elev=30, azim=45)
plt.show()
