import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation

class Globo:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.vidas = 10

    def colisionar(self, otro_globo):
        if self.vidas > 0 and otro_globo.vidas > 0:
            if self.x == otro_globo.x and self.y == otro_globo.y and self.z == otro_globo.z:
                self.vidas -= 1
                otro_globo.vidas -= 1

    def esta_vivo(self):
        return self.vidas > 0


globos = []
globos.append(Globo(0, 0, 0))
globos.append(Globo(1, 1, 1))
globos.append(Globo(2, 2, 2))
globos.append(Globo(3, 3, 3))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter([], [], [], s=1000)  # Tamaño de los globos

def actualizar_animacion(frame):
    ax.cla()

    for globo in globos:
        ax.scatter(globo.x, globo.y, globo.z, c='r', marker='o', s=1000)  # Tamaño de los globos

    for globo in globos:
        for otro_globo in globos:
            if globo != otro_globo:
                globo.colisionar(otro_globo)

    globos[:] = [globo for globo in globos if globo.esta_vivo()]

    ax.set_xlim3d(0, 4)
    ax.set_ylim3d(0, 4)
    ax.set_zlim3d(0, 4)

    sc._offsets3d = ([globo.x for globo in globos], [globo.y for globo in globos], [globo.z for globo in globos])

animacion = animation.FuncAnimation(fig, actualizar_animacion, frames=1, interval=200)  # 1 cuadro

plt.show()
