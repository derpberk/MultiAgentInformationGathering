from re import M
from dolfin import *
import meshio
import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix

def export(assembly,name,vertex_to_dof_map):
    mat = as_backend_type(assembly).mat()
    op = csr_matrix(mat.getValuesCSR()[::-1], shape=mat.size)
    op=op[v2d,:]
    op=op[:,v2d]
    scipy.sparse.save_npz(name+'.npz',op)


N=50

mesh = UnitSquareMesh(N, N)

V = FunctionSpace(mesh, 'CG', 1)
u = TrialFunction(V)
v = TestFunction(V)

v2d=vertex_to_dof_map(V)

#Laplace
A = assemble(inner(grad(u), grad(v)) * dx)
export(A,"laplace",v2d)

#Gradient X
A = assemble(inner(nabla_grad(u)[0], v) * dx)
export(A,"grad_x",v2d)

#Gradient Y
A = assemble(inner(nabla_grad(u)[1], v) * dx)
export(A,"grad_y",v2d)

#Node coordinates
np.save("vertices.npy",mesh.coordinates())

