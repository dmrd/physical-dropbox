from scipy.spatial import Delaunay as delaunay 
import numpy as np
import sys
import matplotlib.pyplot as plt

def plot_triangle(a,b,c):
	x = [a[0],b[0],c[0]]
	y = [a[1],b[1],c[1]]
	z = [a[2],b[2],c[2]]
	verts = [zip(x, y,z)]
	ax.add_collection3d(Poly3DCollection(verts))

n_samples=10**5
data = np.random.uniform(size=3*n_samples).reshape((-1,3))
data=np.loadtxt('cup.csv',dtype='float',delimiter=',')

triangulation = delaunay(data)
n_exterior_faces=(triangulation.neighbors==-1).sum(axis=None)
print n_exterior_faces
exterior_faces=np.zeros(shape=(n_exterior_faces,3),dtype='int')

face_counter=0
for i,tetrahedron in enumerate(triangulation.neighbors):
	for j,face in enumerate(tetrahedron):
		if face ==-1:
			exterior_faces[face_counter,:j]=triangulation.simplices[i,:j]
			exterior_faces[face_counter,j:]=triangulation.simplices[i,j+1:]
			
			face_counter+=1	
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
fig = plt.figure()
ax = Axes3D(fig)

for triangle in exterior_faces:
	plot_triangle(data[triangle[0]],data[triangle[1]],data[triangle[2]])

plt.show()


