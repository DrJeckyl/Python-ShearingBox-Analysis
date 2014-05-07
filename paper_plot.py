from Functions import *
from numpy import *

## Things to Produce ##
# B, b
# U, u
# \nabla \cdot Jh
# <u x b> \cdot <B>
# h_c
# h
# \partial_t h
# < u \cdot \nabla x u >
# [-h_c - < u \cdot \nabla x u >]B^2
# -<u^2> <B \cdot J>

## Take the zaverage and save to disk

#First, load in the file location

dir_path = '/1/home/jackelb/Research/AthenaDumps/AthenaDumps/strat128z4_HF/'
dump = 'Strat.1231.vtk'

#dt's are:
#0.8185432
#0.8150406
#0.8133661
dt = (0.8185432 + 0.8150406)/2. #I'm assuming here that I'll do a central difference (1232 - 1230)/2dt
#dt = 0.8185432

print 'Reading in data from:',dir_path+dump
N, D, B, U, rho = read_data(dir_path + dump, dens_inc=True)

print 'Taking a highpass of B and U'
b = delta(B)
u = delta(U)

save('./PlotData/bav',(zaverage(B),zaverage(b)))
save('./PlotData/uav',(zaverage(U),zaverage(u)))

print 'Defining k'
k = define_k(N,D)

print 'Calculating Magnetic Helicity Flux'
J_H, J_h = MagneticHelicityFlux(dir_path+dump, delt=True)

save('./PlotData/JHav',(zaverage(J_H),zaverage(J_h)))

print 'Taking the divergence of JH and Jh'
divJH = ifftvec( 1.j*dot_p(k, fftvec(J_H)))
divJh = ifftvec( 1.j*dot_p(k, fftvec(J_h)))


save('./PlotData/divJHav',(zaverage(divJH),zaverage(divJh)))

print 'Calculating the scale transfer term'
scale_transfer = dot_p(LargeScale(crossp(u, b)), LargeScale(B))

save('./PlotData/scale_transfer_av',zaverage(scale_transfer))

print 'Calculating the Current Helicity'
Hc, hc = CurrentHelicity(dir_path+dump, delt=True)

save('./PlotData/Hcav',(zaverage(Hc),zaverage(hc)))

print 'Calculating the Magnetic Helicity'
Hm, hm = MagneticHelicity(dir_path+dump, delt=True)

save('./PlotData/Hmav',(zaverage(Hm),zaverage(hm)))

print 'Taking the time derivative of Hm and hm'
H0, h0 = MagneticHelicity(dir_path+'Strat.1230.vtk', delt=True)
H1, h1 = MagneticHelicity(dir_path+'Strat.1232.vtk', delt=True)

dH = (H1-H0)/2./dt
dh = (h1-h0)/2./dt

save('./PlotData/dHav',(zaverage(dH),zaverage(dh)))

print 'Calculating the fluid helicity'
h = LargeScale(helicity(u,k))

save('./PlotData/Hav',zaverage(h))

print 'Calculating the parallel components to the emf'
temp = (-hc - h)*dot_p(B, B)

save('./PlotData/emf_par_av',zaverage(temp))

print 'Calculating the turbulent diffusion term'
diffusion = -LargeScale(dot_p(u, u)) * LargeScale(-Hc)                

save('./PlotData/TDav',zaverage(diffusion))

print 'Disipation:'
diss_s = 2.*eta()*hc

save('./PlotData/dissav',zaverage(diss_s))
