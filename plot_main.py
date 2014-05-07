#!/usr/bin/env python
'''
This routine reads in data from Shanes MHD simulation located in 
/home/ben/Research/DavisSimulation/AthenaDumpsFromShane/ and calculates various quantities
'''
#import numpy,scipy,pylab as np
#import matplotlib as plt
from numpy import *
from scipy import *  
from Functions import *
from matplotlib.pyplot import figure, gcf, plot, title, xlabel, ylabel, tight_layout, savefig, close, loglog
import matplotlib as mpl
from time import time
from numpy.fft import fftfreq

start = time()

mpl.rcParams.update({'font.size':13, 'family':'serif'})
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.linewidth'] = 2
mac_colors = [(122./255.,0./255.,60./255.),(25./255.,57./255.,137./255.)]
mpl.rcParams['axes.color_cycle'] = mac_colors
#Define some TeX strings to use in the legends etc.
JH_TeX = [r'$\nabla \cdot \mathbf{J_H}$', #0) div Jh
          r'$\nabla \cdot \mathbf{J_h}$', #1) div JH
          r'$\nabla \cdot \mathbf{J_{h,alt}}$', #2) div Jh_alt
          r'$\nabla \cdot \mathbf{A} \times \left[ \mathbf{E} + \nabla \Phi \right]$', #3) definition of div JH
          r'$\nabla \cdot \langle \mathbf{a} \times \left[ \mathbf{e} + \nabla \phi \right] \rangle$', #4) definition of div Jh
          r'$\nabla \cdot \left( \mathbf{A} \rangle \times \left[ \langle \mathbf{E} \rangle + \nabla \langle \Phi \rangle \right] \right)$'] #5) definition of div Jh_alt

dynamo_TeX = r'$2  \langle \mathbf{B} \rangle \cdot \langle \mathbf{u} \times \mathbf{b} \rangle$'

res_TeX = [r'$\nabla \cdot \mathbf{A} \times \left[ \mathbf{E} + \nabla \Phi - \eta \mathbf{ \langle J \rangle} \right]$', #0) divJH
           r'$\nabla \cdot \langle \mathbf{a} \times \left[ \mathbf{e} + \nabla \phi -\eta \mathbf{j} \right] \rangle$', #1) div Jh
           r'$\nabla \cdot \left( \langle \mathbf{A} \rangle \times \left[ \langle \mathbf{E} \rangle + \nabla \langle \Phi \rangle - \eta \mathbf{J} \right] \right)$', #2) divJh_alt
           r'$2 \eta \mathbf{\langle J \rangle} \cdot \mathbf{\langle B \rangle}$', #3 resistive term
           r'$2 \eta \langle \mathbf{j} \cdot \mathbf{b} \rangle$'] #4 resistive term


#Override the old TeX definitions for the new JH method
JH_TeX[3] = r'$\nabla \cdot \left( \mathbf{A} \times \mathbf{E} + \Phi \mathbf{B} \right)$'
JH_TeX[4] = r'$\nabla \cdot \left( \langle \mathbf{a} \times \mathbf{e} \rangle + \langle \phi \mathbf{b} \rangle \right)$'
JH_TeX[5] = r'$\nabla \cdot \left( \langle \mathbf{A} \rangle  \times \langle \mathbf{E} \rangle + \langle \Phi \mathbf{B} \rangle \right)$'

res_TeX[0] = r'$\nabla \cdot \left( \mathbf{A} \times \mathbf{E} + \Phi \mathbf{B} - \eta \mathbf{J} \times \mathbf{A} \right)$'
res_TeX[1] = r'$\nabla \cdot \left( \langle \mathbf{a} \times \mathbf{e} \rangle + \langle \phi \mathbf{b} \rangle  + \eta \mathbf{j} \times \mathbf{a} \right)$'
res_TeX[2] = r'$\nabla \cdot \left( \langle \mathbf{A} \rangle  \times \langle \mathbf{E} \rangle + \langle \Phi \mathbf{B} \rangle + \eta \langle \mathbf{J} \rangle \times \langle \mathbf{A} \rangle \right)$'


#Name of the files that are able to be read in
files = [
"Strat.0500.small.vtk", #0
"Strat.0501.small.vtk", #1
"Strat.0502.small.vtk", #2
"Strat.0503.small.vtk", #3
####
"Strat.0020.HF.vtk",    #4
"Strat.0021.HF.vtk",    #5
"Strat.0022.HF.vtk",    #6
"Strat.0023.HF.vtk",    #7
"Strat.0024.HF.vtk",    #8
####
"Strat.0601.big.vtk",   #9
####
"Strat.1180.vtk",       #10
"Strat.1181.vtk",       #11
"Strat.1182.vtk",       #12
"Strat.1183.vtk",       #13
"Strat.1184.vtk",       #14
####
"Strat.0101.vtk",       #15
"Strat.0102.vtk",       #16
"Strat.0103.vtk",       #17
"Strat.0104.vtk",       #18
####
"Strat.1229.vtk",       #19
"Strat.1230.vtk",       #20
"Strat.1231.vtk",       #21
"Strat.1232.vtk",       #22
####
"Strat.0101.vtk",       #23
"Strat.0102.vtk",       #24
"Strat.0103.vtk",       #25
"Strat.0104.vtk",       #26
####
"Strat.1230.vtk",       #27
"Strat.1231.vtk",       #28
"Strat.1232.vtk",       #29
"Strat.1233.vtk"]       #30

#Just some hard coded absolute paths to the data...
sharcnet = False
desktop = True

#Read in precalculated quantities for plotting or remake them from scratch?
remake_quantities = True

if(sharcnet==True):
    file1 = "/work/jackelb/AthenaDumps/"

if(desktop==True):
    file1 = '/1/home/jackelb/Research/AthenaDumps/AthenaDumps/'

file_start = 28
dt = .01*2.*pi/.001  #---- Standard dt
  
if file_start <14:
    dir = '128/OrbitalAverageData/'
if file_start > 14 & file_start < 19:
    dir = 'strat64z6/'
if file_start > 19 & file_start < 23:
    dir = 'strat128z4/'
if file_start > 22 & file_start < 27:
    dir = 'strat64_2/'
    dt = dt/10.
if file_start > 26:
    dir = 'strat128z4_HF/'
    #dt's are:
    #0.8185432
    #0.8150406
    #0.8133661
    #dt = (0.8185432 + 0.8150406)/2. #I'm assuming here that I'll do a central difference (1232 - 1230)/2dt
    dt = 0.8185432
file = file1 + dir + files[file_start]


#Calculate the time derivative of H?
H_dot = True
#H_dot = False


print "Reading in data from", file
print "..."

N, D, B, U, rho = read_data(file,dens_inc=True)
if(remake_quantities == False):
    B = []
    U = []
    B, b = load(file1+dir+"B"+".npy")
    U, u = load(file1+dir+"U"+".npy")
else:
    b = delta(B)
    u = delta(U)

nx, ny, nz = N
dx, dy, dz = D

print "Creating the 3D position matricies and k-vectors..."

k = define_k(N, D)
x = linspace(-.5, .5, nx)
y = linspace(-2., 2., ny)
zz = linspace(-2., 2., nz)

kz = fftfreq(nz,dz)*2.*pi

###### Plot b ######
Plot_Title = [R'$\langle B_x \rangle$', R'$\langle B_y \rangle$', R'$\langle B_z \rangle$']
xL = ['z', 'z', 'z']
yL = [R'$\langle B_x \rangle$', R'$\langle B_y \rangle$', R'$\langle B_z \rangle$']
Plot_Title = Plot_Title + [R'$\langle b_x \rangle$', R'$\langle b_y \rangle$', 
                           R'$\langle b_z \rangle$']
xL = xL + ['z', 'z', 'z']
yL = yL + [r'$b_x$', r'$b_y$', r'$b_zb$' ]
filename = "../Plots/b.pdf"
plot_2vec(zz,  zaverage(B-b), zaverage(b), Plot_Title, xL, yL, filename)

save('./PlotData/Bav.np',(zaverage(B),zaverage(b)))

figure
loglog(range(nz/2), power_spectrum(B[0])+power_spectrum(B[1])+power_spectrum(B[2]))
title(r'Power spectrum of $\| \mathbf{B} \|$')
ylabel(r' $ \| \tilde{\mathbf{B}} \| $')
xlabel(r'$k_z$')
tight_layout()
savefig("../Plots/B_psd.pdf")
close()

figure
loglog(range(nz/2), weighted_power_spectrum(B[0],kz)+weighted_power_spectrum(B[1],kz)+weighted_power_spectrum(B[2],kz))
title(r'Power spectrum of $\| \mathbf{B} \|$')
ylabel(r' $ \| \tilde{\mathbf{B}} \| $')
xlabel(r'$k_z$')
tight_layout()
savefig("../Plots/B_psd_kz.pdf")
close()

BL=LargeScale(B)

figure
loglog(range(nz/2), weighted_power_spectrum(B[0],kz)+weighted_power_spectrum(B[1],kz)+weighted_power_spectrum(B[2],kz),range(nz/2), weighted_power_spectrum(b[0],kz)+weighted_power_spectrum(b[1],kz)+weighted_power_spectrum(b[2],kz),range(nz/2), weighted_power_spectrum(BL[0],kz)+weighted_power_spectrum(BL[1],kz)+weighted_power_spectrum(BL[2],kz))
title(r'Power spectrum of $\| \mathbf{B} \|$')
ylabel(r' $ \| \tilde{\mathbf{B}} \| $')
xlabel(r'$k_z$')
tight_layout()
savefig("../Plots/B_psd_kz_test_filter.pdf")
close()

#Define E
#E = v x b
print "EMF, v x b ..."
if(remake_quantities):
    E, e = EMF(file, U, B, delt=True)
else:
    E, e = load(file1+dir+'E'+".npy")


###### Plot E ######
Plot_Title = [R'$\langle E_x \rangle$', R'$\langle E_y \rangle$', R'$\langle E_z \rangle$']
xL = ['z', 'z', 'z']
yL = [r'$\langle E_x \rangle$', r'$\langle E_y \rangle$', r'$\langle E_z \rangle$' ]
Plot_Title = Plot_Title + [R'$e_x$', R'$e_y$', R'$e_z$']
xL = xL + ['z', 'z', 'z']
yL = yL + [r'$e_x$', r'$e_y$', r'$e_z$' ]
filename = "../Plots/E.pdf"
plot_2vec(zz,  zaverage(E-e), zaverage(e), Plot_Title, xL, yL, filename)

save('./PlotData/Eav.np',(zaverage(E),zaverage(e)))

###### Plot u ######
Plot_Title = [R'$\langle U_x \rangle$', R'$\langle U_y \rangle$', R'$\langle U_z \rangle$']
xL = ['z', 'z', 'z']
yL = [r'$\langle U_x \rangle$', r'$\langle U_y \rangle$', r'$\langle U_z \rangle$' ]
Plot_Title = Plot_Title + [R'$u_x$', R'$u_y$', R'$u_z$']
xL = xL + ['z', 'z', 'z']
yL = yL + [r'$u_x$', r'$u_y$', r'$u_z$' ]
filename = "../Plots/u.pdf"
plot_2vec(zz,  zaverage(U-u), zaverage(u), Plot_Title, xL, yL, filename)

save('./PlotData/Uav.np',(zaverage(U),zaverage(u)))

figure
loglog(range(nz/2),  power_spectrum(U[0])+power_spectrum(U[1])+power_spectrum(U[2]))
title(r'Power spectrum of $\| \mathbf{U} \|$')
ylabel(r' $ \| \tilde{\mathbf{U}} \| $')
xlabel(r'$k_z$')
tight_layout()
savefig("../Plots/U_psd.pdf")
close()

figure
loglog(range(nz/2), weighted_power_spectrum(U[0],kz)+weighted_power_spectrum(U[1],kz)+weighted_power_spectrum(U[2],kz))
title(r'Power spectrum of $\| \mathbf{U} \|$')
ylabel(r' $ \| \tilde{\mathbf{U}} \| $')
xlabel(r'$k_z$')
tight_layout()
savefig("../Plots/U_psd_k.pdf")
close()

U = []

print "Calculating the current density, J..."
if(remake_quantities):
    J, jc = CurrentDensity(file, B, k, delt=True)
else:
    J, jc = load(file1+dir+'J'+".npy")

#Tubulent diffusion term
TD = LargeScale(dot_p(u,u) * dot_p(B,J))
save('./PlotData/TD',zaverage(TD))

#### Dynamo Term ####
dissipation_s = 2.*eta()*LargeScale(dot_p(jc,b))
dissipation_l = 2.*eta()*(dot_p((J-jc),(B-b)))       

if(remake_quantities==False):
    divjh_compare = load(file1+dir+'H_t.npy')
else:
    #divjh_compare = -dot_p( 2.*delta(B), delta(E-e))
    #divjh_compare = -2.*dot_p( delta(B), delta(cross_p(u,b))) - dissipation
    divjh_compare = 2.*dot_p( LargeScale(B), LargeScale(crossp(u,b)))
    divjh_compare_alt = 2.*LargeScale(dot_p(B, crossp(u,b)))
    #b = []
    #u = []



Plot_Title = [R'$\langle J_x \rangle$', R'$\langle J_y \rangle$', R'$\langle J_z \rangle$']
xL = ['z', 'z', 'z']
yL = [r'$\langle J_x \rangle$', r'$\langle J_y \rangle$', r'$\langle J_z \rangle$' ]
Plot_Title = Plot_Title + [R'$j_x$', R'$j_y$', R'$j_z$']
xL = xL + ['z', 'z', 'z']
yL = yL + [r'$j_x$', r'$j_y$', r'$j_z$' ]
filename = "../Plots/J.pdf"
plot_2vec(zz,  zaverage(J-jc), zaverage(jc), Plot_Title, xL, yL, filename)

save('./PlotData/Jav.np',(zaverage(J),zaverage(jc)))
#jc = []

#Define the current helicity
#j_dot_b = j dot b
print "The current helicity j dot b ..."
if(remake_quantities):
    Hc, hc = CurrentHelicity(file, J, B, delt=True)
else:
    Hc, hc = load(file1 + dir+'Hc'+".npy")

## figure
## plot(zz,  zaverage(Hc - hc), '+')
## title('Large Scale Current helicity')
## ylabel(r' $\langle \mathbf{J} \cdot \mathbf{B} \rangle$')
## xlabel('z')
## tight_layout()
## savefig("../Plots/Hc.pdf")
## close()

figure
plot(zz,  zaverage(delta(hc)), '+')
title('Small scale current helicity')
ylabel(r'$\langle \mathbf{j} \cdot \mathbf{b} \rangle$')
xlabel('z')
tight_layout()
savefig("../Plots/hc.pdf")
close()
save('./PlotData/Hcav.np',(zaverage(Hc),zaverage(hc)))
Hc = []
#hc = []


print "Vector and scalar potentials ..."
if(remake_quantities):
    Pot_4, pot_4 = Potential(file, J, E, k, delt=True)
else:
    Pot_4, pot_4 = load(file1 + dir+'Potential'+".npy")

Phi = Pot_4[0]
phi = pot_4[0]
A = Pot_4[1:]
a = pot_4[1:]

###### Plot a ######
Plot_Title = [R'$\langle A_x \rangle$', R'$\langle A_y \rangle$', R'$\langle A_z \rangle$']
xL = ['z', 'z', 'z']
yL = [r'$\langle A_x \rangle$', r'$\langle A_y \rangle$', r'$\langle A_z \rangle$' ]
Plot_Title = Plot_Title + [R'$a_x$', R'$a_y$', R'$a_z$']
xL = xL + ['z', 'z', 'z']
yL = yL + [r'$a_x$', r'$a_y$', r'$a_z$' ]
filename = "../Plots/a.pdf"
plot_2vec(zz, zaverage(A-a), zaverage(a), Plot_Title, xL, yL, filename)
save('./PlotData/Aav.np',(zaverage(A),zaverage(a)))
a = []
figure
plot(zz, zaverage(delta(Phi)), '+')
title('Large Scale Scalar potential')
ylabel(r' $\langle \Phi \rangle$')
xlabel('z')
tight_layout()
savefig("../Plots/Phi.pdf")
close()
save('./PlotData/Phiav.np',zaverage(Phi))

#magnetic helicity flux, jh= a x [E + grad(phi)]
print "Magnetic helicity flux, F ..."
if(remake_quantities):
    J_H, J_h, JH_large = MagneticHelicityFlux(file, Pot_4, E, delt=True, large = True)
else:
    J_H, J_h, JH_large = load(file1+dir+'JH'+".npy")


    
divJH = ifftvec( 1.j*dot_p(k, fftvec(J_H)))
divJh = ifftvec( 1.j*dot_p(k, fftvec(J_h)))
divJH_large = ifftvec( 1.j*dot_p(k, fftvec(JH_large)))

####
# Plot JH
Plot_Title = [R'$\langle \mathbf{J_H}_x \rangle$', R'$\langle \mathbf{J_H}_y \rangle$', 
              R'$\langle \mathbf{j_h}_z \rangle$']
xL = ['z', 'z', 'z']
yL = [r'$\langle \mathbf{J_H}_x \rangle$', r'$\langle \mathbf{J_H}_y \rangle$', 
      r'$\langle \mathbf{J_H}_z \rangle$' ]
filename = "../Plots/JH.pdf"
plot_vec(zz, zaverage(J_H), Plot_Title, xL, yL, filename)
close()
save('./PlotData/JHav.np',(zaverage(J_H),zaverage(J_h)))

if(remake_quantities):
    figure
    pt = gcf().add_subplot(111)
    lines = pt.plot(zz,  zaverage(J_H[0]), '+', zz, zaverage(JH_large[0]), 'ro', zz, zaverage(J_h[0]), 'b.')
    title('Magnetic helicity flux components')
    xlabel('z')
    tight_layout()
    lgd = pt.legend(lines, [r'$\mathbf{J_{H,total}}$', r'$\langle \mathbf{J_H} \rangle$', 
                            r'$\langle \mathbf{J_h} \rangle$'], 
                    loc = 'upper center', bbox_to_anchor = (0.5, -0.1))
    savefig("../Plots/JH_components.pdf", bbox_extra_artists=(lgd, ), bbox_inches='tight')
    close()

#Calculate magnetic helicty, a.dot.b
print "Magnetic helicity, a dot b ..."
if(remake_quantities):
    H, h = MagneticHelicity(file, Pot_4, B, delt=True)
else:
    H, h = load(file1+dir+'H'+".npy")

figure
p1 = gcf().add_subplot(111)
lines = p1.plot(zz, zaverage(hc), '-', zz, max(zaverage(hc))/max(zaverage(h))*zaverage(h), '-')
title(r'$\langle \mathbf{j} \cdot \mathbf{b} \rangle$' + ' versus' + r'$\langle \mathbf{a} \cdot \mathbf{b} \rangle$'  )
xlabel('z')
#Add the resistivity terms to the legends
if(eta() > 0.):
    lgd = p1.legend(lines, [r'$\langle \mathbf{j} \cdot \mathbf{b} \rangle$', r'$\langle \mathbf{a} \cdot \mathbf{b} \rangle$'], loc = 'upper center', bbox_to_anchor = (0.5, -0.1))
else:
    lgd = p1.legend(lines, [r'$\langle \mathbf{j} \cdot \mathbf{b} \rangle$', r'$\langle \mathbf{a} \cdot \mathbf{b} \rangle$'], loc = 'upper center', bbox_to_anchor = (0.5, -0.1))

tight_layout()
savefig("../Plots/helicity_compare.pdf", bbox_extra_artists = (lgd, ), bbox_inches='tight')
close()

figure
p1 = gcf().add_subplot(111)
lines = p1.plot(zz, zaverage(-hc-LargeScale(helicity(u,k))*dot_p(B,B)), '-', zz, zaverage(LargeScale(-divjh_compare/2.)), '-')
title(r'$\langle \mathbf{j} \cdot \mathbf{b} \rangle B^2$' + r' versus ' + r'$\langle \mathbf{u} \times \mathbf{b} \rangle \cdot \mathbf{B}$'  )
xlabel('z')
#Add the resistivity terms to the legends
if(eta() > 0.):
    lgd = p1.legend(lines, [r'$\langle \mathbf{j} \cdot \mathbf{b} \rangle B^2$', r'$\langle \mathbf{u} \times \mathbf{b} \rangle$ \cdot \mathbf{B}$'], loc = 'upper center', bbox_to_anchor = (0.5, -0.1))
else:
    lgd = p1.legend(lines, [r'$\langle \mathbf{j} \cdot \mathbf{b} \rangle B^2$', r'$\langle \mathbf{u} \times \mathbf{b} \rangle \cdot \mathbf{B}$'], loc = 'upper center', bbox_to_anchor = (0.5, -0.1))

tight_layout()
savefig("../Plots/emf_current_helicity.pdf", bbox_extra_artists = (lgd, ), bbox_inches='tight')
close()

figure
plot(zz, zaverage(H-h), '+')
title('Large Scale Magnetic Helicity')
ylabel(r' $\langle \mathbf{A} \cdot \mathbf{B} \rangle$')
xlabel('z')
tight_layout()
savefig("../Plots/H.pdf")
close()

figure
plot(zz, zaverage(h), '+')
title('Small scale magnetic helicity')
ylabel(r'$\langle \mathbf{a} \cdot \mathbf{b} \rangle$')
xlabel('z')
tight_layout()
savefig("../Plots/h_small.pdf")
close()
save('./PlotData/Hav.np',(zaverage(H),zaverage(h)))

print "Calculating div(F) and 2b dot (v x b) ..."


#divjh_compare = -dot_p( 2.*delta(B), delta(E-e))- dissipation

figure
p1 = gcf().add_subplot(111)
lines = p1.plot(zz, zaverage(divJh), '-', zz, -zaverage(divjh_compare) - zaverage(dissipation_s), '-')
title(JH_TeX[0] + ' versus' +  dynamo_TeX )
xlabel('z')
#Add the resistivity terms to the legends
if(eta() > 0.):
    lgd = p1.legend(lines, [res_TeX[1], r'$-$'+dynamo_TeX+r'$+$'+res_TeX[4]], loc = 'upper center', bbox_to_anchor = (0.5, -0.1))
else:
    lgd = p1.legend(lines, [JH_TeX[4], r'$-$'+dynamo_TeX], loc = 'upper center', bbox_to_anchor = (0.5, -0.1))

tight_layout()
savefig("../Plots/divJh-compare.pdf", bbox_extra_artists = (lgd, ), bbox_inches='tight')
close()
save('./PlotData/divjhav.np',zaverage(divJh))
save('./PlotData/divjh_compareav.np',zaverage(divjh_compare))

if H_dot:

    #        dh, dH = dh_dt([file1+'128/'+files[i] for i in [10,12]],dt)
    
    #dh, dH = dh_dt([file1+dir+files[i] for i in [file_start-1,file_start+1]],dt)
    dh, dH = dh_dt([file1+dir+files[i] for i in [file_start-1,file_start]],dt,back=True)
    save('./PlotData/dHdtav.np',(zaverage(dH),zaverage(dh)))

    #dH = (H2_T - H0_T)/2./dt
    #dh = (h2 - h0)/2./dt
    #dH_l = (H2 - H0)/2./dt

    figure
    plot(zz, zaverage(LargeScale(dH)), '+')
    title('Time derivative of the magnetic helicity at t=0021')
    xlabel('z')
    ylabel(r'$\partial_t \langle H \rangle $')
    tight_layout()
    savefig("../Plots/dHdt.pdf")
    close()

    figure
    pt = gcf().add_subplot(111)
    #gca().set_color_cycle(mac_colors)
    lines = pt.plot(zz, zaverage(dh), zz, -zaverage(divJh) - zaverage(divjh_compare) - zaverage(dissipation_s))
    title('Small scale magnetic helicity flux')
    xlabel('z')
    tight_layout()
    #Add the resistivity terms to the legends
    if(eta() > 0.):
        lgd = pt.legend(lines, [r'$\partial_t \langle \mathbf{a \cdot b} \rangle$', r'$-$'+res_TeX[1]+r'$+$'+dynamo_TeX+r'$+$'+res_TeX[4]], 
                    loc = 'upper center', bbox_to_anchor = (0.5, -0.1))    
    else:
        lgd = pt.legend(lines, [r'$\partial_t \mathbf{h}$', r'$-$'+JH_TeX[4]+r'$-$'+dynamo_TeX], 
                    loc = 'upper center', bbox_to_anchor = (0.5, -0.1))
	lgd.get_frame().set_alpha(0.0)				
    savefig("../Plots/small_scale_h_evolution.pdf", bbox_extra_artists=(lgd, ), bbox_inches='tight', pad_inches=0,transparent=True)
    close()

    figure
    pt = gcf().add_subplot(111)
    #lines = pt.plot(zz, zaverage(dh), zz, -zaverage(divJh) - zaverage(LargeScale(divjh_compare)) - zaverage(LargeScale(dissipation_s)))
    lines = pt.plot(zz,zaverage(dh+LargeScale(divJh)),zz,zaverage(LargeScale(-divjh_compare-dissipation_s)))
    title('small scale divJh and Total-Large scale divJH')
    xlabel('z')
    tight_layout()
    #Add the resistivity terms to the legends
    if(eta() > 0.):
        lgd = pt.legend(lines, [r'$\partial_t \langle \mathbf{a \cdot b} \rangle$', r'$-$'+res_TeX[1]+r'$+$'+dynamo_TeX+r'$+$'+res_TeX[4]], 
                    loc = 'upper center', bbox_to_anchor = (0.5, -0.1))    
    else:
        lgd = pt.legend(lines, [JH_TeX[4], JH_TeX[3] + r'$-$' + JH_TeX[5]], 
                    loc = 'upper center', bbox_to_anchor = (0.5, -0.1))
    savefig("../Plots/testing1.pdf", bbox_extra_artists=(lgd, ), bbox_inches='tight')
    close()

    figure
    pt = gcf().add_subplot(111)
    #lines = pt.plot(zz, zaverage(), zz, zaverage(LargeScale()))
    lines = pt.plot(zz, zaverage(LargeScale(dot_p(jc,b)*dot_p(B,B))-TD*1000.), '-', zz, zaverage(LargeScale(-divjh_compare/2. )), '-')
    title('Time slice derivatives and RHS')
    xlabel('z')
    tight_layout()
    #Add the resistivity terms to the legends
    if(eta() > 0.):
        lgd = pt.legend(lines, [r'$\partial_t \langle \mathbf{a \cdot b} \rangle$', r'$-$'+res_TeX[1]+r'$+$'+dynamo_TeX+r'$+$'+res_TeX[4]], 
                    loc = 'upper center', bbox_to_anchor = (0.5, -0.1))    
    else:
        lgd = pt.legend(lines, [r'$\partial_t \mathbf{h}$', r'$-$'+JH_TeX[4]+r'$-$'+dynamo_TeX], 
                    loc = 'upper center', bbox_to_anchor = (0.5, -0.1))
    savefig("../Plots/testing2.pdf", bbox_extra_artists=(lgd, ), bbox_inches='tight')
    close()

    figure
    pt = gcf().add_subplot(111)
    lines = pt.plot(zz, zaverage(dh), zz, -zaverage(divJh))
    title('Time slice derivatives and RHS')
    xlabel('z')
    tight_layout()
    #Add the resistivity terms to the legends
    if(eta() > 0.):
        lgd = pt.legend(lines, [r'$\partial_t \langle \mathbf{a \cdot b} \rangle$', r'$-$'+res_TeX[1]+r'$+$'+dynamo_TeX+r'$+$'+res_TeX[4]], 
                    loc = 'upper center', bbox_to_anchor = (0.5, -0.1))    
    else:
        lgd = pt.legend(lines, [r'$\partial_t \mathbf{h}$', r'$-$'+JH_TeX[4]+r'$-$'+dynamo_TeX], 
                    loc = 'upper center', bbox_to_anchor = (0.5, -0.1))
    savefig("../Plots/dhdt_vs_divjh.pdf", bbox_extra_artists=(lgd, ), bbox_inches='tight')
    close()



    ## figure
    ## pt = gcf().add_subplot(111)
    ## lines = pt.plot(zz, zaverage(dH_l), '-', zz, -zaverage(divJH_large) + zaverage(divjh_compare)-zaverage(dissipation_s), '-')
    ## title('Time slice derivatives and RHS')
    ## xlabel('z')
    ## tight_layout()
    ## #Add the resistivity terms to the legends
    ## if(eta() > 0.):
    ##     lgd = pt.legend(lines, [r'$\partial_t \left( \langle \mathbf{A} \rangle \cdot \langle \mathbf{B} \rangle \right)$', r'$-$'+res_TeX[2]+r'$+$'+dynamo_TeX+r'$+$'+res_TeX[3]], 
    ##                 loc = 'upper center', bbox_to_anchor = (0.5, -0.1))    
    ## else:
    ##     lgd = pt.legend(lines, [r'$\partial_t  \left( \langle \mathbf{A} \rangle \cdot \langle \mathbf{B} \rangle \right)$', r'$-$'+JH_TeX[5]+r'$+$'+dynamo_TeX], 
    ##                 loc = 'upper center', bbox_to_anchor = (0.5, -0.1))
    
    ## savefig("../Plots/large_scale_h_evolution.pdf", bbox_extra_artists=(lgd, ), bbox_inches='tight')
    ## close()

    figure
    pt = gcf().add_subplot(111)
    #gca().set_color_cycle(mac_colors)
    lines = pt.plot(zz, zaverage(LargeScale(dH)), zz, -zaverage(LargeScale(divJH)) + zaverage(dissipation_l))
    title('Total Magnetic Helicity Conservation')
    xlabel('z')
    tight_layout()
    #Add the resistivity terms to the legends
    if(eta() > 0.):
        lgd = pt.legend(lines, [r'$\partial_t \langle H \rangle$', r'$-$'+res_TeX[0]+r'$+$'+res_TeX[3]],
                    loc = 'upper center', bbox_to_anchor = (0.5, -0.1))
    else:
        lgd = pt.legend(lines, [r'$\partial_t \langle H \rangle$', r'$-$'+JH_TeX[3]],
                    loc = 'upper center', bbox_to_anchor = (0.5, -0.1))
    lgd.get_frame().set_alpha(0.0)
    savefig("../Plots/total_H_evolution.pdf", bbox_extra_artists=(lgd, ), bbox_inches='tight', pad_inches=0,transparent=True)
    close()

    figure
    pt = gcf().add_subplot(111)
    lines = pt.plot(zz,  zaverage(H), '+', zz, zaverage(H-h), 'ro', zz, zaverage(h), 'b.')
    title('Magnetic helicity components')
    xlabel('z')
    tight_layout()
    lgd = pt.legend(lines, [r'$\mathbf{H}_{total}$', r'$\langle \mathbf{H} \rangle$', 
                            r'$\langle \mathbf{h} \rangle$'], 
                    loc = 'upper center', bbox_to_anchor = (0.5, -0.1))
    savefig("../Plots/H_components.pdf", bbox_extra_artists=(lgd, ), bbox_inches='tight')
    close()

print 'Total elapsed time:',time()-start
