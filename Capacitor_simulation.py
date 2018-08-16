#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 15:38:28 2018

@author: haiyan
"""
"""
2 1 "core_metal"
2 2 "metal_air"
2 6 "boundary"
2 7 "boundary_in"
2 8 "boundary_out"
2 9 "core_air"
3 3 "air"
3 4 "core"
3 5 "metal"
"""


from fenics import *
import numpy
#set_log_level (ERROR)
#pwd = "/home/osmanabu/Downloads/Assignment_3"
mesh = Mesh('Mesh_8.xml')
cells = MeshFunction('size_t', mesh, 'Mesh_8_physical_region.xml')
facets = MeshFunction('size_t', mesh, 'Mesh_8_facet_region.xml')
file_Mesh = File("mesh_facets.pvd")
file_Mesh << facets 
exit()

def material_coefficient(target_mesh, cells_list, coeffs):
    coeff_func = Function(FunctionSpace(target_mesh, 'DG', 0))
    markers = numpy.asarray(cells_list.array(), dtype=numpy.int32)
    coeff_func.vector() [:] = numpy.choose(markers-3, coeffs)
    return coeff_func

n = FacetNormal (mesh)
#interface , area, volume elements
di = Measure('dS', domain=mesh, subdomain_data=facets)
da = Measure('ds', domain=mesh, subdomain_data=facets)
dv = Measure('dx', domain=mesh, subdomain_data=cells)

scalar = FiniteElement('P', tetrahedron, 1)
vector = VectorElement('P', tetrahedron, 1)
Vector = VectorFunctionSpace(mesh, 'P', 1)
#Tensor = TensorElement('P', tetrahedron, 1)
mixed_element = MixedElement([scalar, vector])
Space = FunctionSpace(mesh, mixed_element)

#units: m, kg, s, A, V, K
delta = Identity(3)
levicivita3 = as_tensor([( (0,0,0),(0,0,1),(0,-1,0) ), ( (0,0,-1),(0,0,0),(1,0,0) ), ( (0,1,0),(-1,0,0),(0,0,0)) ])
epsilon = levicivita3

eps_0 = 8.85E-12 #in A s/(V m)
mu_0 = 12.6E-7 #in V s/(A m)

null=1E-20

#air
varsigma_air = 3E-15
chi_el_air = null
chi_ma_air = null
mu_r_ma_air = chi_ma_air +1.

#Copper
varsigma_cu = 58.5E+6 #in S/m or in 1/(Ohm m)
chi_el_cu = null
chi_ma_cu = -1E-5
mu_r_ma_cu = chi_ma_cu +1.

#Teflon (ptfe) is an insulator
varsigma_ptfe = 1E-25
chi_el_ptfe = 1.0
chi_ma_ptfe = 1E-6
mu_r_ma_ptfe = chi_ma_ptfe + 1.

chi_el = material_coefficient(mesh, cells, [chi_el_air, chi_el_cu, chi_el_ptfe])
chi_ma = material_coefficient(mesh, cells, [chi_ma_air, chi_ma_cu, chi_ma_ptfe])
mu_r_ma = material_coefficient(mesh, cells, [mu_r_ma_air, mu_r_ma_cu, mu_r_ma_ptfe])
varsigma = material_coefficient(mesh, cells, [varsigma_air, varsigma_cu, varsigma_ptfe])

tMax = 0.5
Dt = 0.1
t = 0.0

capacitor_1 = Expression('5.*time',degree=1 ,time=0)
capacitor_2 = Expression('-5.*time',degree=1 ,time=0)
bc01=DirichletBC(Space.sub(0), capacitor_1, facets, 7)
bc02=DirichletBC(Space.sub(0), capacitor_2, facets, 8)
bc03=DirichletBC(Space.sub(0), Constant(0.), facets, 6)
bc04=DirichletBC(Space.sub(1), Constant((0.,0.,0.)), facets, 6)

bc = [bc01, bc02, bc03, bc04]

dunkn = TrialFunction(Space)
test = TestFunction(Space)
del_phi, del_A = split(test)

unkn = Function(Space)
unkn0 = Function(Space)
unkn00 = Function(Space)

unkn_init = Expression(('0.0', '0.0', '0.0', '0.0'), degree=1)
unkn00 = interpolate(unkn_init, Space)
unkn0.assign(unkn00)
unkn.assign(unkn0)

phi, A = split(unkn)
phi0, A0 = split(unkn0)
phi00, A00 = split(unkn00)

i,j,k,l = indices(4)
delta = Identity(3)
E = as_tensor(-phi.dx(i)-(A-A0)[i]/Dt, (i,))
E0 = as_tensor(-phi0.dx(i)-(A0-A00)[i]/Dt, (i,))
B = as_tensor(epsilon[i,j,k]*A[k].dx(j), (i,))

D = eps_0*E
D0 = eps_0*E0
H = 1./mu_0*B
P = eps_0*chi_el*E
P0 = eps_0*chi_el*E0
mD = D + P
mD0 = D0 + P0
MM = 1./mu_0/mu_r_ma*chi_ma*B
J_fr = varsigma*E

F_phi = (-(mD-mD0)[i]*del_phi.dx(i) - Dt*J_fr[i]*del_phi.dx(i) - Dt*epsilon[i,j,k]*MM[k].dx(j)*del_phi.dx(i) )\
        *(dv(3)+dv(4)+dv(5)) + ( n('+')[i]*Dt*(J_fr('+')-J_fr('-'))[i]*del_phi('+')+n('+')[i]*Dt*epsilon[i,j,k]*\
          (MM('+')[k].dx(j)- MM('-')[k].dx(j))*del_phi('+') )*(di(1)+di(2)+di(9) )

F_A = (eps_0*(A-2.*A0+A00)[i]/Dt/Dt*del_A[i] + 1./mu_0*A[i].dx(j)*del_A[i].dx(j)\
       -J_fr[i]*del_A[i] - (P-P0)[i]/Dt*del_A[i] + epsilon[i,j,k]*MM[k]*del_A[i].dx(j)\
       )*(dv(3)+dv(4)+dv(5))

Form = F_phi + F_A
Gain = derivative(Form, unkn, dunkn)

file_phi_metal = File('phi_metal.pvd')
file_phi_ptfe = File('phi_ptfe.pvd')
file_E = File('E.pvd')
file_B = File('B.pvd')

mesh_metal = SubMesh(mesh, cells, 5)
mesh_ptfe = SubMesh(mesh, cells, 4)

VectorSpace_metal = FunctionSpace(mesh_metal, 'P', 1)
VectorSpace_ptfe = FunctionSpace(mesh_ptfe, 'P', 1)
phi_metal = Function(VectorSpace_metal, name='$\phi$')
phi_ptfe = Function(VectorSpace_ptfe, name='$\phi$')


while t<tMax:
    t += Dt
    print ('time: ',t)
    capacitor_1.time = t
    capacitor_2.time = t
    solve(Form==0, unkn, bc, J=Gain, \
          solver_parameters={"newton_solver":{"linear_solver":\
                            "mumps", "relative_tolerance": 1e-5} },\
            form_compiler_parameters={"cpp_optimize": True, "representation":"uflacs", "quadrature_degree": 2} )
    
    phi_metal.assign(project(unkn.split(deepcopy=True)[0], VectorSpace_metal))
    file_phi_metal << (phi_metal, t)
    phi_ptfe.assign(project(unkn.split(deepcopy=True)[0], VectorSpace_ptfe))
    file_phi_ptfe << (phi_ptfe, t)
    file_B << (project(B,Vector),t)
    file_E << (project(E,Vector),t)
    
    unkn00.assign(unkn0)
    unkn0.assign(unkn)







