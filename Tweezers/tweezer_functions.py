import math

from scipy import constants
import numpy as np
from IonChainTools import calcPositions,lengthScale
from scipy.optimize import fsolve
import itertools
from itertools import product
import pandas as pd
import re

#Constants in SI units
eps0 = constants.epsilon_0
m = 39.9626*constants.atomic_mass
c = constants.c
e = constants.e
hbar = constants.hbar
pi = np.pi


def potential(omega_tweezer,linewidths,omega_res,P_opt,beam_waist):
    '''
    Find the dipole potential of the optical tweezers beam for
    the given set of parameters at r=0 and z=0 -- without RWA


    omega_tweezer = angular frequency of tweezer laser beam [2*Pi x Hz]
    linewidths = linewidth of the given resonant transition of ion taken from NIST database in angular frequency units 
        (ex, S1/2 to P1/2 and S1/2 to P3/2 for 40Ca+), this is a list/array of some form with all relevent transitions
    P_opt = total optical power of tweezer laser beam
    omega_res = angular frequency of resonant transition, also based off NIST data [2*Pi x Hz]
    beam_waists = beamwaist of the tweezer laser beam
    given its frequency and the NA of our system or from measurement
    '''
    p = []
    for i in range(len(linewidths)): 
        p.append( (-3.*P_opt*(c**2.)/((omega_res[i]**3.)*(beam_waist**2.))) * (linewidths[i]/((omega_res[i] - omega_tweezer)) +
                                          linewidths[i]/(omega_res[i] + omega_tweezer)) ) 
    pot = sum(p)
    return pot


def potentialRWA(omega_tweezer,linewidths,omega_res,P_opt,beam_waist):
    '''
    Find the potential of the optical tweezers beam for
    the given set of parameters at r=0 and z=0 -- with RWA


    omega_tweezer = angular frequency of tweezer laser beam [2*Pi x Hz]
    linewidths = linewidth of the given resonant transition of ion taken from NIST database in angular frequency units 
        (ex, S1/2 to P1/2 and S1/2 to P3/2 for 40Ca+), this is a list/array of some form with all relevent transitions
    P_opt = total optical power of tweezer laser beam
    omega_res = angular frequency of resonant transition, also based off NIST data [2*Pi x Hz]
    beam_waists = beamwaist of the tweezer laser beam
    given its frequency and the NA of our system or from measurement
    '''
    p = []
    for i in range(len(linewidths)): 
        p.append( (-3*P_opt*(c**2)/((omega_res[i]**3)*(beam_waist**2))) * (linewidths[i]/((omega_res[i] - omega_tweezer)))  ) 
    pot = sum(p)
    return pot


def scattering(omega_tweezer,linewidths,omega_res,P_opt,beam_waist):
    '''
    Find the scattering of the optical tweezers beam (at r=0 and z=0) off of a given resonance
    for the given set of parameters -- without RWA
    omega_tweezer = angular frequency of tweezer laser beam [2*Pi x Hz]
    linewidths = linewidth of the given resonant transition taken from NIST database in angular frequency units 
        (ex, S1/2 to P1/2 and S1/2 to P3/2 for 40Ca+)
    P_opt = total optical power of tweezer laser beam
    omega_res = angular frequency of resonant transition, also based off NIST data [2*Pi x Hz]
    beam_waists = beamwaist of the tweezer laser beam
    given its frequency and the NA of our system or from measurement
    '''
    s = []
    for i in range(len(linewidths)):
        s.append(((3*(c**2)*P_opt)/(hbar *pi* (omega_res[i]**3)*(beam_waist**2))) *((omega_tweezer/omega_res[i])**3)* (((linewidths[i]/(omega_res[i] - omega_tweezer))+
                                                                            (linewidths[i]/(omega_res[i] + omega_tweezer)))**2) )
    scat = sum(s)
    return scat


def scatteringRWA(omega_tweezer,linewidths,omega_res,P_opt,beam_waist):
    '''
    Find the scattering of the optical tweezers beam (at r=0 and z=0) off of a given resonance
    for the given set of parameters -- with RWA
    omega_tweezer = angular frequency of tweezer laser beam [2*Pi x Hz]
    linewidths = linewidth of the given resonant transition taken from NIST database in angular frequency units 
        (ex, S1/2 to P1/2 and S1/2 to P3/2 for 40Ca+)
    P_opt = total optical power of tweezer laser beam
    omega_res = angular frequency of resonant transition, also based off NIST data [2*Pi x Hz]
    beam_waists = beamwaist of the tweezer laser beam
    given its frequency and the NA of our system or from measurement
    '''
    s = []
    for i in range(len(linewidths)):
        s.append( (3*P_opt*(c**2)/(hbar *pi* (omega_res[i]**3)*(beam_waist**2))) *((omega_tweezer/omega_res[i])**3) *((linewidths[i]/(omega_res[i] - omega_tweezer))**(2)) 
        )
    scat = sum(s)
    return scat

def rayleigh_length(w0, lambda_beam):
    """
    Calculate the Rayleigh length of a beam
    inputs: 
    w0 -- minimum beam waist [m]
    lambda_beam -- wavelength of the beam in meters [m]
    output:
    Rayleigh length in meters
    """
    return (pi * w0**2) / lambda_beam

def beam_propogation(w0, z_pos, lambda_beam):
    """
    Calculate the beam propagation 
    inputs:
    w0 --  minimum beam waist [m]
    z_pos -- list of z positions [m]
    lambda_beam -- wavelength of the beam [m]
    
    output:
    beam propagation as a function of z
    """
    rayleigh = rayleigh_length(w0, lambda_beam)
    return w0 * np.sqrt(1 + (z_pos / rayleigh)**2)

def intensity(x_pos,y_pos,P0, wz):
    """
    Calculate the intensity of a Gaussian beam at a given (r,z)
    inputs:
    x_pos -- list of x positions [m]
    y_pos -- list of y positions [m]
    P0 -- power [W]
    wz -- beam waist [m] calculated from beam_propogation function
    
    returns:
    Intensity in W/m^2
    """
    return (2 * P0 / (np.pi * wz**2))  * np.exp(-2 * x_pos**2 / wz**2) * np.exp(-2 * y_pos**2 / wz**2)

def intensity_TEM10(x_pos,y_pos,w0,wz,E0,n):
    """
    Calculate the intensity of a TEM10 beam at a given (r,z)
    inputs:
    x_pos -- list of x positions [m]
    y_pos -- list of y positions [m]
    P0 -- power [W]
    w0 -- minimum beam waist at z=0 [m], same one that is used in beam_propogation function
    wz -- beam waist [m] calculated from beam_propogation function

    returns:
    Intensity in W/m^2
    """
    return ( (( c * eps0 * n / 2 ) * (E0 **2 * w0**2 ))/ (wz**2) )* (8*x_pos**2 / (wz**2)) * np.exp(-2*x_pos**2 / wz**2) * np.exp(-2*y_pos**2 / wz**2)

def half_angle_beam_divergence(M_squared,w0,lambda_beam):
    """
    Calculate the half angle beam divergence
    inputs:
    M_squared -- beam quality factor
    w0 -- beam waist [m]
    lambda_beam -- wavelength of the beam [m]
    returns:
    half angle beam divergence in radians
    """
    return M_squared * lambda_beam/ (pi* w0)

def potential_position_dependent(omega_res,linewidths,omega_tweezer,intensity):
    """
    Calculate the potential of the optical tweezer beam at a specific position (r,z)
    
    
    
    """
    #rayleigh = rayleigh_length(FWHM,lambda_beam) 
    #beam_prop = beam_propogation(FWHM,rayleigh_length,z_pos)  
    #intens = intensity(P0,FWHM,beam_propogation,r)

    p = []
    for i in range(len(linewidths)): 
        p.append( (-3.*pi*(c**2.)/(2*(omega_res[i]**3.))) \
                                       * (linewidths[i]/((omega_res[i] - omega_tweezer)) +
                                          linewidths[i]/(omega_res[i] + omega_tweezer)) * intensity )
    pot = sum(p)
    return pot

def scattering_position_dependent(omega_res, linewidths, omega_tweezer, intensity):
    s = []
    for i in range(len(linewidths)):
        s.append(((3 * c**2 * intensity) / (2 * hbar * omega_res[i]**3)) *
                 (omega_tweezer / omega_res[i])**3 * 
                 ((linewidths[i] / (omega_res[i] - omega_tweezer) +
                   linewidths[i] / (omega_res[i] + omega_tweezer))**2))
    scat = sum(s)
    return scat
'''
def pot_derivative_with_tweeze(x, omega_rf_axial, omega_tw_radial, tweezed_ion, displacement):
    """
    derivative of the potential energy of the ion chain, use this to find positions of ions in the trap
    This one is specifically for tweezing one ion in the chain
    inputs:

    x: list of ion positions
    omega_rf_axial: the rf axial trapping frequency [Hz]
    omega_tw_radial: the tweezer radial trapping frequency [Hz]
    tweezed_ion: Ion number for the tweezed ion MAKE ME A LIST
    displacement: distance between the tweezer beam center and position of the tweezed ion MAKE ME A LIST
    """
    N = len(x)
    A = 1/2 * m * omega_rf_axial**2
    B = (e**2) /(4 * pi * eps0)
    C = 1/2 * m * omega_tw_radial**2
    
    return [A*(x[m]) 
            - sum([B / (abs(x[m] - x[n])**2) for n in range(m) if x[m] != x[n]])  # Avoid division by zero
            + sum([B / (abs(x[m] - x[n])**2) for n in range(m+1, N) if x[m] != x[n]])  # Avoid division by zero
            #MAKE ME A LOOP SO THE LIST MAKES SENSE
            + (C*(x[m] - displacement) if m == tweezed_ion else 0)  # Only apply tweezer potential to the tweezed ion
            for m in range(N)]
'''

'''
def pot_derivative_with_tweeze(x, omega_rf_axial, omega_tw_radial, tweezed_ion, displacement):
    """
    derivative of the potential energy of the ion chain, use this to find positions of ions in the trap
    This one is specifically for tweezing one ion in the chain
    inputs:

    x: list of ion positions
    omega_rf_axial: the rf axial trapping frequency [Hz]
    omega_tw_radial: the tweezer radial trapping frequency [Hz]
    tweezed_ion: Ion number for the tweezed ion MAKE ME A LIST
    displacement: distance between the tweezer beam center and position of the tweezed ion MAKE ME A LIST
    """
    N = len(x)
    A = 1/2 * m * omega_rf_axial**2
    B = (e**2) /(4 * pi * eps0)
    C = 1/2 * m * omega_tw_radial**2
    
    return [A*(x[m]) 
            - sum([B / (abs(x[m] - x[n])**2) for n in range(m) if x[m] != x[n]])  # Avoid division by zero
            + sum([B / (abs(x[m] - x[n])**2) for n in range(m+1, N) if x[m] != x[n]])  # Avoid division by zero
            #MAKE ME A LOOP SO THE LIST MAKES SENSE
            + (C*(x[tweezed_ion] - displacement) if m == tweezed_ion else 0)  # Only apply tweezer potential to the tweezed ion
            for m in range(N)]
'''



def pot_derivative_with_tweeze(x, omega_rf_axial, omega_tw_radial, tweezed_ion, displacement):
    """
    derivative of the potential energy of the ion chain, use this to find positions of ions in the trap
    This one is specifically for tweezing one ion in the chain
    inputs:

    x: list of ion positions
    omega_rf_axial: the rf axial trapping frequency [Hz]
    omega_tw_radial: the tweezer radial trapping frequency [Hz]
    tweezed_ion: Ion number for the tweezed ion MAKE ME A LIST
    displacement: distance between the tweezer beam center and position of the tweezed ion MAKE ME A LIST
    """
    N = len(x)
    A = m * omega_rf_axial**2
    B = (e**2) /(4 * pi * eps0)
    C = m * omega_tw_radial**2
    
    return [A*(x[m]) 
            - sum([B / (abs(x[m] - x[n])**2) for n in range(m) if x[m] != x[n]])  # Avoid division by zero
            + sum([B / (abs(x[m] - x[n])**2) for n in range(m+1, N) if x[m] != x[n]])  # Avoid division by zero
            #MAKE ME A LOOP SO THE LIST MAKES SENSE
            + (C*(x[tweezed_ion] - displacement) if m == tweezed_ion else 0)  # Only apply tweezer potential to the tweezed ion
            for m in range(N)]







'''
def pot_derivative_dimensionless(
    x,
    omega_rf_axial,
    omega_tw_radial,
    tweezed_ion,
    displacement
):
    N = len(x)

    alpha = (omega_tw_radial / omega_rf_axial)**2

    return [
        x[m]
        - sum(1 / (x[m] - x[n])**2 for n in range(m))
        + sum(1 / (x[n] - x[m])**2 for n in range(m+1, N))
        + (alpha * (x[m] - displacement) if m == tweezed_ion else 0)
        for m in range(N)
    ]



def pot_derivative_SI(x, omega_rf_axial, omega_tw_radial, tweezed_ion, displacement):
    N = len(x)

    A = m * omega_rf_axial**2
    B = (e**2) / (4 * np.pi * eps0)
    C = m * omega_tw_radial**2

    return [
        A * x[m]
        - sum(B / (x[m] - x[n])**2 for n in range(m))
        + sum(B / (x[m] - x[n])**2 for n in range(m+1, N))
        + (C * (x[m] - displacement) if tweezed_ion is not None and m == tweezed_ion else 0)
        for m in range(N)
    ]
'''










def pot_derivative_with_2tweeze(x, omega_rf_axial, omega_tw_radial, tweezed_ion1,tweezed_ion2, displacement1,displacement2):
    """
    derivative of the potential energy of the ion chain, use this to find positions of ions in the trap
    This one is specifically for tweezing one ion in the chain
    inputs:

    x: list of ion positions
    omega_rf_axial: the rf axial trapping frequency [2*pi*Hz]
    omega_tw_radial: the tweezer radial trapping frequency [2*pi*Hz]
    tweezed_ion: Ion number for the tweezed ion
    displacement: distance between the tweezer beam center and position of the tweezed ion
    """
    N = len(x)
    A = 1/2 * m * omega_rf_axial**2
    B = (e**2) /(4 * pi * eps0)
    C = 1/2 * m * omega_tw_radial**2
    
    return [A*(x[m]) 
            - sum([B / (abs(x[m] - x[n])**2) for n in range(m) if x[m] != x[n]])  # Avoid division by zero
            + sum([B / (abs(x[m] - x[n])**2) for n in range(m+1, N) if x[m] != x[n]])  # Avoid division by zero
            + C*(x[tweezed_ion1] + displacement1) if m == tweezed_ion1 else 0  # Only apply tweezer potential to the tweezed ion1
            + C*(x[tweezed_ion2] - displacement2) if m == tweezed_ion2 else 0  # Only apply tweezer potential to the tweezed ion2|
            for m in range(N)]

def ion_spacing(N,omega_a):
    """
    Calculating the equilibrium positions of the ions in real units, as well as the distance between each ion
    inputs:
    N = number of ions
    omega_a = rf axial trap frequency [2*Pi x Hz]

    returns:
    list where first entry is list of equilibrium positions of ions in meters and second entry is list of distances between ions in meters
    """
    A = np.zeros((N, N))
    l = lengthScale(omega_a)
    ueq = calcPositions(N)*l
    
    diff_list = []
    for x, y in zip(ueq[0::], ueq[1::]):
        diff_list.append(y-x)
    return [ueq,diff_list]




def ion_spacing_tweezers(potential_from_tweezers,ionspacing,omega_rf_axial,omega_tw_radial,tweezed_ion,displacement):
    ueq = fsolve(potential_from_tweezers,ionspacing[0],args = (omega_rf_axial,omega_tw_radial,tweezed_ion,displacement))
    diff_list = []
    for x, y in zip(ueq[0::], ueq[1::]):
        diff_list.append(y-x)
    return [ueq,diff_list]

def ion_spacing_2_tweezers(pot_derivative_with_2tweeze,ionspacing,omega_rf_axial, omega_tw_radial, tweezed_ion1,tweezed_ion2, displacement1,displacement2):
    ueq = fsolve(pot_derivative_with_2tweeze,ionspacing[0],args = (omega_rf_axial, omega_tw_radial, tweezed_ion1,tweezed_ion2, displacement1,displacement2))
    diff_list = []
    for x, y in zip(ueq[0::], ueq[1::]):
        diff_list.append(y-x)
    return [ueq,diff_list]

def omega_tweezer_r(U,beam_waist,m):
    """
    Calculating the radial tweezer trap frequency (perpendicular to laser propogation) at r=0 and z=0
    given the tweezer potential U [J],
    the beam waist of the tweezer laser beam,
    and the mass of the ion

    U = potential created from the tweezer laser beam, from potential function at r=0 and z=0
    beam_waists = beam_waists = beamwaist of the tweezer laser beam
    m = mass of ion
    """
    return ((abs(U) * 4) / (m * (beam_waist)**2))**(1/2)

def omega_tweezer_a(U,beam_waist,tweezer_wavelength,m):
    """
     Calculating the axial tweezer trap frequency (along laser propogation) at r=0 and z=0
    given the tweezer potential U [J],
    the beam waist of the tweezer laser beam,
    and the mass of the ion

       U = potential created from the tweezer laser beam, from potential function
       beam_waists = beam_waists = beamwaist of the tweezer laser beam
       tweezer_wavelength = wavelgth of the tweezer laser beam
       m = mass of ion
       """
    return ((2*abs(U)/m)**(1/2)) * 1/((pi*(beam_waist**2)/tweezer_wavelength))

def TEM10_tweezer_optical_potential_to_trap_frequency_y(linewidths, omega_res,omega_tweezer, w0, m, P0):
    p = []
    for i in range(len(linewidths)): 
        p.append(np.sqrt(
            ((8*P0/(np.exp(1)*pi*w0**4))*((-2*pi*c**2) * (2/m) ) /(2*omega_res**3) ) * (linewidths[i]/((omega_res[i] - omega_tweezer)) +
                                            linewidths[i]/(omega_res[i] + omega_tweezer))
        ))
    pot = sum(p)
    return pot

def TEM10_tweezer_optical_potential_to_trap_frequency_x(linewidths, omega_res,omega_tweezer, w0, m, P0):
    p = []
    for i in range(len(linewidths)): 
        p.append(np.sqrt(
            ((40*P0/(np.exp(1)*pi*w0**4))*((-2*pi*c**2) * (2/m) )/(m*omega_res[i]**3*w0**2) * (linewidths[i]/((omega_res[i] - omega_tweezer)) +
                                            linewidths[i]/(omega_res[i] + omega_tweezer)))
        ))
    pot = sum(p)
    return pot

def mode_calc_r(m,omega_r_combined,ueq,N):
    
    """
    
    Hessian for ions in a pseudo-potential
    
    Inputs:
    N: number of ions 
    ueq -- list of equilibrium positions of ions (m)
    m -- mass of ion (kg)
    omega_r_combined -- combined radial frequency taking into account the rf potential as well as the tweezer potentials. 
                will look like array where each entry for untweezed ion is the rf radial frequency and each entry for the
                tweezed ions is sqrt(omega_tweezer^2 + omega_r_rf^2) (2*pi*Hz)
    omega_a -- axial trapping frequency created by rf potential (2*pi*Hz)
    
    Outputs: 
    modes -- list of tuples where each tuple is a mode (frequency [Hz], eigenvector)
    """
    A = np.zeros((N, N))
    coloumb = ((e**2) / (4 * pi * eps0))
    masses = np.array([m for _ in range(N)])
    for i in range(N):
        A[i][i] = (masses[i] * omega_r_combined[i]**2 - coloumb * sum(1 / (ueq[i] - ueq[m])**3 for m in range(0, i))
           - coloumb * sum(1 / (ueq[m] - ueq[i])**3 for m in range(i + 1, N)))# * masses[i]
        for j in range(0, i):
            A[i][j] = (1/(ueq[i]-ueq[j])**3) *(coloumb)#* np.sqrt(masses[i])*np.sqrt(masses[j])
        for j in range(i+1, N):
            A[i][j] = (1/ (ueq[j]-ueq[i])**3) *(coloumb)#*np.sqrt(masses[i])*np.sqrt(masses[j])
    eigvals, eigvecs = np.linalg.eig(A) # this gives eigenvalues and eigenvectors
    freqs =( np.sqrt(1*eigvals/m))/(2*pi) #eigenvalue = spring constant k, so freq = sqrt(e-val)/(2*pi*m)
    
    
    scaledmodes = [(f, v) for f, v in zip(freqs, eigvecs.T)]
    scaledmodes = sorted(scaledmodes, key=lambda mode: mode[0],reverse=True)
    modes = []
    for f, scaledvec in scaledmodes:
        vec = np.array([scaledvec[i]/1 for i in range(len(eigvals))])
        vec = vec / np.sqrt(vec.dot(vec))
        modes.append((f, vec))
    return modes

def mode_calc_a(m,omega_a_combined,ueq,N):
    """
    Hessian for ions in a pseudo-potential
    Inputs:
    ueq -- equilibrium positions of the ions (m)
    m -- mass of ion (kg)
    omega_a_combined -- combined radial frequency taking into account the rf potential as well as the tweezer potentials. 
                will look like array where each entry for untweezed ion is the rf radial frequency and each entry for the
                tweezed ions is sqrt(omega_tweezer^2 + omega_a_rf^2) (2*pi*Hz)
    omega_a -- axial trapping frequency created by rf potential (2*pi*Hz)
    
    Outputs: 
    modes -- list of tuples where each tuple is a mode (frequency [Hz], eigenvector)
    """

    A = np.zeros((N, N))
    coloumb = ((e**2) / (4 * pi * eps0))
    masses = np.array([m for _ in range(N)])
    for i in range(N):
        A[i][i] = (masses[i] * omega_a_combined[i]**2 + coloumb * sum(2 / (ueq[i] - ueq[m])**3 for m in range(0, i))
           + coloumb * sum(2 / (ueq[m] - ueq[i])**3 for m in range(i + 1, N)))# * masses[i]
        for j in range(0, i):
            A[i][j] = (-2/(ueq[i]-ueq[j])**3) *(coloumb)#* np.sqrt(masses[i])*np.sqrt(masses[j])
        for j in range(i+1, N):
            A[i][j] = (-2/ (ueq[j]-ueq[i])**3)*(coloumb)# *np.sqrt(masses[i])*np.sqrt(masses[j])

    eigvals, eigvecs = np.linalg.eig(A) # this gives eigenvalues and eigenvectors
    freqs =( np.sqrt(1*eigvals/m))/(2*pi) #eigenvalue = spring constant k, so freq = sqrt(e-val)/(2*pi*m)

    scaledmodes = [(f, v) for f, v in zip(freqs, eigvecs.T)]
    scaledmodes = sorted(scaledmodes, key=lambda mode: mode[0],reverse=False)
    modes = []
    for f, scaledvec in scaledmodes:
        vec = np.array([scaledvec[i]/1 for i in range(len(eigvals))])
        vec = vec / np.sqrt(vec.dot(vec))
        modes.append((f, vec))
    return modes

def eta(mode_structure,qubit_wavelength,N):
    """input:
    mode structure as output from mode_calc_r or mode_calc_a
    N = number of ions
    qubit_wavelength = wavelength of qubit transition [m] (729e-9 for Ca)
    output:
    eta values for each mode and ion instead of just the eigenvectors
    """

    eta = []
    for mode in mode_structure:
        eta.append([mode[1][i] * (2 * pi / qubit_wavelength) * np.sqrt(hbar / (2 * m * mode[0])) for i in range(N)])
    return eta

'''
def combined_frequencies(N,tweezed_ions,w_tweezer_r,w_tweezer_a,w_rf_r,w_rf_a):
    
    takes in rf and tweezer trap frequencies and adds together frequencies in quadruture
    radial modes will be effected by either the radial and axial tweezer directions (in BladeRunner setup)
    axial modes will be effected by only tweezer radial
    
    inputs:
    N = number of ions
    tweezed_ions = list of which ions are getting tweezed
    w_tweezer_r = radial trapping frequency of tweezer [2*Pi x Hz]
    w_tweezer_a = axial trapping frequency of tweezer [2*Pi x Hz]
    w_rf_r = radial rf trapping frequency [2*Pi x Hz]
    w_rf_a = axial rf trapping frequency [2*Pi x Hz]
    
    returns: array of potential combined trapping frequencies
                [0] is radial rf and radial tweezer
                [1] is radial rf and axial tweezer
                [2] is axial rf and radial tweezer
    
    

    omeg_tweezer_r = np.zeros(N)
    omeg_tweezer_a = np.zeros(N)
    omeg_tweezer_r[tweezed_ions] = w_tweezer_r
    omeg_tweezer_a[tweezed_ions] = w_tweezer_a

    omeg_rf_r = w_rf_r * np.ones(N) 
    omeg_rf_a = w_rf_a * np.ones(N)

    omega_combined_rr = np.sqrt(omeg_rf_r**2 + omeg_tweezer_r**2)
    omega_combined_ra = np.sqrt(omeg_rf_r**2 + omeg_tweezer_a**2)
    omega_combined_ar = np.sqrt(omeg_rf_a**2 + omeg_tweezer_r**2)
    
    return np.array([omega_combined_rr,omega_combined_ra,omega_combined_ar])
'''

def combined_frequencies(N,tweezed_ions,w_tweezer_r,w_tweezer_a,w_rf_r,w_rf_a):
    '''
    takes in rf and tweezer trap frequencies and adds together frequencies in quadruture
    radial modes will be effected by either the radial and axial tweezer directions (in BladeRunner setup)
    axial modes will be effected by only tweezer radial
    
    inputs:
    N = number of ions
    tweezed_ions = list of which ions are getting tweezed
    w_tweezer_r = radial trapping frequency of tweezer [2*Pi x Hz]
    w_tweezer_a = axial trapping frequency of tweezer [2*Pi x Hz]
    w_rf_r = radial rf trapping frequency [2*Pi x Hz]
    w_rf_a = axial rf trapping frequency [2*Pi x Hz]
    
    returns: array of potential combined trapping frequencies
                [0] is radial rf and radial tweezer
                [1] is radial rf and axial tweezer
                [2] is axial rf and radial tweezer
    
    '''

    omeg_tweezer_r = np.zeros(N)
    omeg_tweezer_a = np.zeros(N)
    omeg_tweezer_r[tweezed_ions] = w_tweezer_r
    omeg_tweezer_a[tweezed_ions] = w_tweezer_a

    omeg_rf_r = w_rf_r * np.ones(N) 
    omeg_rf_a = w_rf_a * np.ones(N)

    omega_combined_rr = np.sqrt(omeg_rf_r**2 + omeg_tweezer_r**2)
    omega_combined_ra = np.sqrt(omeg_rf_r**2 + omeg_tweezer_a**2)
    omega_combined_ar = np.sqrt(omeg_rf_a**2 + omeg_tweezer_r**2)
    omega_combined_aa = np.sqrt(omeg_rf_a**2 + omeg_tweezer_a**2)
    
    return np.array([omega_combined_rr,omega_combined_ra,omega_combined_ar,omega_combined_aa])


def trapping_ratios(w_tweezer_r,w_tweezer_a,w_rf_r,w_rf_a):
    
    """
    takes in tweezer and rf trap frequencies and outputs various ratios of frequencies in case this turns out to be helpful
    
    inputs:
    w_tweezer_r = radial trapping frequency of tweezer [2*Pi x Hz]
    w_tweezer_a = axial trapping frequency of tweezer [2*Pi x Hz]
    w_rf_r = radial rf trapping frequency [2*Pi x Hz]
    w_rf_a = axial rf trapping frequency [2*Pi x Hz]
    
    returns: array of ratios of omegas
    """
    
    tweezer_r_to_rf_ratio = w_tweezer_r / w_rf_r
    tweezer_r_to_axial_ratio = w_tweezer_r / w_rf_a
    tweezer_a_to_rf_ratio = w_tweezer_a / w_rf_a
    
    return np.array([tweezer_r_to_rf_ratio,tweezer_r_to_axial_ratio])


def individual_freqs_to_mode_vectors(N,tweezed_ions,w_tweezer_r,w_tweezer_a,w_rf_r,w_rf_a,ueq):
    """
    takes in output from combined_frequencies and outputs new radial modes
    
    inputs:
    N = number of ions
    tweezed_ions = list of which ions are getting tweezed
    w_tweezer_r = radial trapping frequency of tweezer [2*Pi x Hz]
    w_tweezer_a = axial trapping frequency of tweezer [2*Pi x Hz]
    w_rf_r = radial rf trapping frequency [2*Pi x Hz]
    w_rf_a = axial rf trapping frequency [2*Pi x Hz]
    
    returns: modes from mode_calc_r
    
    """
    combined_freqs = combined_frequencies(N,tweezed_ions,w_tweezer_r,w_tweezer_a,w_rf_r,w_rf_a)
    omega_r_combined = combined_freqs[0]
    #omega_a = w_rf_a
    return mode_calc_r(m,omega_r_combined,ueq,N)

def individual_freqs_to_mode_vectors_axial(N,tweezed_ions,w_tweezer_r,w_tweezer_a,w_rf_r,w_rf_a,ueq):
    """
    takes in output from combined_frequencies and outputs new axial modes
    
    inputs:
    N = number of ions
    tweezed_ions = list of which ions are getting tweezed
    w_tweezer_r = radial trapping frequency of tweezer [2*Pi x Hz]
    w_tweezer_a = axial trapping frequency of tweezer [2*Pi x Hz]
    w_rf_r = radial rf trapping frequency [2*Pi x Hz]
    w_rf_a = axial rf trapping frequency [2*Pi x Hz]
    
    returns: modes from mode_calc_r
    
    """
    combined_freqs = combined_frequencies(N,tweezed_ions,w_tweezer_r,w_tweezer_a,w_rf_r,w_rf_a)
    ### CHANGED THIS FROM [2] to [3] BECAUSE I ADDED IN THE COMBINED AA FREQUENCY IN THE COMBINED FREQUENCIES FUNCTION, CHECK THIS CAREFULLY ###
    omega_a_combined = combined_freqs[2]
    #omega_a = w_rf_a
    return mode_calc_a(m,omega_a_combined,ueq,N)

def individual_freqs_to_mode_vectors_radial_weak(N,tweezed_ions,w_tweezer_r,w_tweezer_a,w_rf_r,w_rf_a):
    """
    I'm unsure what this function even is...
    takes in output from combined_frequencies and outputs new radial modes (but with the weak trappping from tweezers)
    
    inputs:
    N = number of ions
    tweezed_ions = list of which ions are getting tweezed
    w_tweezer_r = radial trapping frequency of tweezer [2*Pi x Hz]
    w_tweezer_a = axial trapping frequency of tweezer [2*Pi x Hz]
    w_rf_r = radial rf trapping frequency [2*Pi x Hz]
    w_rf_a = axial rf trapping frequency [2*Pi x Hz]
    
    returns: modes from mode_calc_r
    
    """
    combined_freqs = combined_frequencies(N,tweezed_ions,w_tweezer_r,w_tweezer_a,w_rf_r,w_rf_a)
    omega_r_combined = combined_freqs[1]
    omega_a = w_rf_a
    return mode_calc_r(m,omega_r_combined,omega_a)

def tweezer_optical_potential_to_trap_frequency(tweezer_wavelength,linewidths,omega_res,P_opt,beam_waist,m,U):
    """
    takes in physical parameters of calcium ion and tweezer beam and outputs expected tweezer trap frequency
    
    Inputs:
    tweezer_wavelength = tweezer wavelength [m]
    linewidths = linewidth of the given resonant transition taken from NIST database in angular frequency units 
        (ex, S1/2 to P1/2 and S1/2 to P3/2 for 40Ca+)
    P_opt = total optical power of tweezer laser beam
    omega_res = angular frequency of resonant transition, also based off NIST data [2*Pi x Hz]
    beam_waist = beamwaist of the tweezer laser beam
    given its frequency and the NA of our system or from measurement
    U = potential(omega_tweezer,linewidths,omega_res,P_opt,beam_waist) OR 
        potential_position_dependent(omega_res,linewidths,omega_tweezer,intensity)
    
    outputs:
    array of radial and axial tweezer trap frequencies [2*Pi x Hz]
    
    """
    omega_tweezer = 2*pi*c/tweezer_wavelength
    
    
    w_tweezer_r =  omega_tweezer_r(U,beam_waist,m)
    w_tweezer_a = omega_tweezer_a(U,beam_waist,tweezer_wavelength,m)
    return np.array([w_tweezer_r,w_tweezer_a])



def physical_params_to_radial_mode_vectors(N,ueq,tweezed_ions,tweezer_wavelength,linewidths,omega_res,w_rf_a,w_rf_r,P_opt,beam_waist,m,U):
    """
    takes in physical parameters of tweezer beam and calcium ion as well as rf 
    trapping parameters to output combined radial modes
    
    Inputs:
    tweezer_wavelength = tweezer wavelength [m]
    linewidths = linewidth of the given resonant transition taken from NIST database in angular frequency units 
        (ex, S1/2 to P1/2 and S1/2 to P3/2 for 40Ca+)
    P_opt = total optical power of tweezer laser beam
    omega_res = angular frequency of resonant transition, also based off NIST data [2*Pi x Hz]
    beam_waist = beamwaist of the tweezer laser beam
    given its frequency and the NA of our system or from measurement  
    N = number of ions
    tweezed_ions = list of which ions are getting tweezed
    w_tweezer_r = radial trapping frequency of tweezer [2*Pi x Hz]
    w_tweezer_a = axial trapping frequency of tweezer [2*Pi x Hz]
    w_rf_r = radial rf trapping frequency [2*Pi x Hz]
    w_rf_a = axial rf trapping frequency [2*Pi x Hz]
    U = potential(omega_tweezer,linewidths,omega_res,P_opt,beam_waist) OR 
        potential_position_dependent(omega_res,linewidths,omega_tweezer,intensity)
    
    outputs:
    modes from mode_calc_r, frequencies in Hz (not angular)
    
    """
   
    omega_tweezer = 2*pi*c/tweezer_wavelength
    
    #U = potential(omega_tweezer,linewidths,omega_res,P_opt,beam_waist)
    w_tweezer_r =  omega_tweezer_r(U,beam_waist,m)
    w_tweezer_a = omega_tweezer_a(U,beam_waist,tweezer_wavelength,m)
    return individual_freqs_to_mode_vectors(N,tweezed_ions,w_tweezer_r,w_tweezer_a,w_rf_r,w_rf_a,ueq)

def physical_params_to_axial_mode_vectors(N,ueq,tweezed_ions,tweezer_wavelength,linewidths,omega_res,w_rf_a,w_rf_r,P_opt,beam_waist,m,U):
    """
    takes in physical parameters of tweezer beam and calcium ion as well as rf 
    trapping parameters to output combined axial modes
    
    Inputs:
    tweezer_wavelength = tweezer wavelength [m]
    linewidths = linewidth of the given resonant transition taken from NIST database in angular frequency units 
        (ex, S1/2 to P1/2 and S1/2 to P3/2 for 40Ca+)
    P_opt = total optical power of tweezer laser beam
    omega_res = angular frequency of resonant transition, also based off NIST data [2*Pi x Hz]
    beam_waist = beamwaist of the tweezer laser beam
    given its frequency and the NA of our system or from measurement  
    N = number of ions
    tweezed_ions = list of which ions are getting tweezed
    w_tweezer_r = radial trapping frequency of tweezer [2*Pi x Hz]
    w_tweezer_a = axial trapping frequency of tweezer [2*Pi x Hz]
    w_rf_r = radial rf trapping frequency [2*Pi x Hz]
    w_rf_a = axial rf trapping frequency [2*Pi x Hz]
    U = potential(omega_tweezer,linewidths,omega_res,P_opt,beam_waist) OR 
        potential_position_dependent(omega_res,linewidths,omega_tweezer,intensity)
    
    outputs:
    modes from mode_calc_r, frequencies in Hz (not angular)
    
    """
   
    omega_tweezer = 2*pi*c/tweezer_wavelength
    
    #U = potential(omega_tweezer,linewidths,omega_res,P_opt,beam_waist)
    w_tweezer_r =  omega_tweezer_r(U,beam_waist,m)
    w_tweezer_a = omega_tweezer_a(U,beam_waist,tweezer_wavelength,m)
    return individual_freqs_to_mode_vectors_axial(N,tweezed_ions,w_tweezer_r,w_tweezer_a,w_rf_r,w_rf_a,ueq)

def physical_params_to_radial_mode_vectors_weak(N,tweezed_ions,tweezer_wavelength,linewidths,omega_res,w_rf_a,w_rf_r,P_opt,beam_waist,m,U):
    """
    unsure what this one is too
    takes in physical parameters of tweezer beam and calcium ion as well as rf 
    trapping parameters to output combined radial modes (from weak tweezer trap)
    
    Inputs:
    tweezer_wavelength = tweezer wavelength [m]
    linewidths = linewidth of the given resonant transition taken from NIST database in angular frequency units 
        (ex, S1/2 to P1/2 and S1/2 to P3/2 for 40Ca+)
    P_opt = total optical power of tweezer laser beam
    omega_res = angular frequency of resonant transition, also based off NIST data [2*Pi x Hz]
    beam_waist = beamwaist of the tweezer laser beam
    given its frequency and the NA of our system or from measurement  
    N = number of ions
    tweezed_ions = list of which ions are getting tweezed
    w_tweezer_r = radial trapping frequency of tweezer [2*Pi x Hz]
    w_tweezer_a = axial trapping frequency of tweezer [2*Pi x Hz]
    w_rf_r = radial rf trapping frequency [2*Pi x Hz]
    w_rf_a = axial rf trapping frequency [2*Pi x Hz]
    U = potential(omega_tweezer,linewidths,omega_res,P_opt,beam_waist) OR 
        potential_position_dependent(omega_res,linewidths,omega_tweezer,intensity)
    
    outputs:
    modes from mode_calc_r, frequencies in Hz (not angular)
    
    """
   
    omega_tweezer = 2*pi*c/tweezer_wavelength
    
    #U = potential(omega_tweezer,linewidths,omega_res,P_opt,beam_waist)
    w_tweezer_r =  omega_tweezer_r(U,beam_waist,m)
    w_tweezer_a = omega_tweezer_a(U,beam_waist,tweezer_wavelength,m)
    return individual_freqs_to_mode_vectors_radial_weak(N,tweezed_ions,w_tweezer_r,w_tweezer_a,w_rf_r,w_rf_a)

#this section down here is functions for the sideband cooling calculations

def tweezer_combos_full_radial(
    omega_tweezer,
    linewidths,
    omega_res,
    m,
    mode_calc_r,
    N_list,
    f_rf_r,
    f_rf_a,
    P_opt,
    w0,
    max_tweezed=1,
    qubit_lambda = 729e-9
):
    """
    Inputs:
    omega_tweezer = optical tweezer beam angular frequency [2*Pi x Hz]
    linewidths = linewidth of the given resonant transition taken from NIST database in angular frequency units 
        (ex, S1/2 to P1/2 and S1/2 to P3/2 for 40Ca+) 
    omega_res = angular frequency of resonant transition, also based off NIST data [2*Pi x Hz]
    m = mass of ion [kg]
    mode_calc_r = function from above to calculate radial modes
    N_list = list of number of ions to loop over (can also be a single integer)
    f_rf_r = radial rf trapping frequency [Hz]
    f_rf_a = axial rf trapping frequency [Hz]
    P_opt = list of total optical power of tweezer laser beam to loop over (can also be a single number) [W]
    w0 = beamwaist of the tweezer laser beam [m] 
    qubit_lambda = wavelength of qubit transition [m] (729e-9 for 40Ca+)

    Outputs:
    DataFrame with columns:
    - N: number of ions
    - Tweezed ions: tuple of which ions are tweezed
    - P_per_tweezer (W): power per tweezer in this configuration
    - Combined radial frequencies: array of combined radial frequencies for each ion in this configuration
    - Mode{i}_freq: frequency of mode i in this configuration (NaN if mode i does not exist for this N)
    - Mode{i}_eigvec: eigenvector of mode i in this configuration (NaN array if mode i does not exist for this N)
    """


    # Making sure N and P are lists because the rest of the code needs them to be lists
    if np.isscalar(N_list):
        N_list = [int(N_list)]
    if np.isscalar(P_opt):
        P_opt = [P_opt]

    #defining pi as just pi because I use it a lot 
    pi = np.pi
    rows = []
    # compute k from provided qubit wavelength 
    k = 2.0 * pi / qubit_lambda
    eigvec_scale = k * np.sqrt(hbar / (2.0 * m))

    # Looping over number of ions, N
    for N in N_list:
        # Generate all possible tweezer combinations, but only over the first N/2 ions
        # Tweezing ions in the second half of the chain should be symmetric to the first half
        half_range = N // 2+1
        # MAke sure max_tweezed does not exceed available positions in the N/2 range
        max_tweezed_local = min(max_tweezed, half_range)
        all_combos = []
        # Create the possible subsets of ions tweezed.  
        # max_tweezed = 1 so this normally just places the tweezer beam at every position
        # in the N/2 range
        for r in range(0, max_tweezed_local + 1):
            all_combos.extend(itertools.combinations(range(half_range), r))

        # Setting up the rf-harmonic trap parameters for later
        w_rf_r = f_rf_r * 2 * pi
        w_rf_r_list = np.full(N, w_rf_r)
        ueq = ion_spacing(N, 2*pi*f_rf_a)[0]

        # Looping over optical tweezer powers, P_opt
        for P_total in P_opt:
            #looping over where the tweezer placement is, determined above in all_combos
            for tweezed_positions in all_combos:
                # If max_tweezed =/= 1, this divides total power equally amongst every tweezer
                n_tweezed = len(tweezed_positions)
                P_per = P_total / n_tweezed if n_tweezed > 0 else 0.0

                # Compute tweezer potential and tweezer trap frequency for this configuration
                pot = potential(omega_tweezer, linewidths, omega_res, P_per, w0)
                w_tw_r = omega_tweezer_r(pot, w0, m)

                # Combine tweezed and untweezed radial frequencies
                # Here we are combining the tweezer radial frequency with the rf radial frequency
                combo = np.array([
                    np.sqrt(w_tw_r**2 + w_rf_r_list[i]**2) if i in tweezed_positions else w_rf_r_list[i]
                    for i in range(N)
                ])

                # Finding radial modes from the combined frequencies, and the e-vecs and e-values
                modes = mode_calc_r(m, combo, ueq, N)
                freqs = np.array([f for f, v in modes], dtype=float) if len(modes) else np.array([], dtype=float)
                if len(modes):
                    eigvecs = np.vstack([np.ravel(v) for f, v in modes])  # shape (n_modes, N)
                else:
                    eigvecs = np.empty((0, N))

                # Build the dataframe for the output of this configuration
                row = {
                    "N": N,
                    "Tweezed ions": tweezed_positions,
                    "P_per_tweezer (W)": P_per,
                    "Combined radial frequencies": combo,
                }

                # Making sure the overall dataframe has the same number of columns for each N
                # Fill Mode{i}_freq and Mode{i}_eigvec for i in [0, N-1]
                for mode_index in range(N):
                    # creating columns for all possible modes, if that N doesn't have those modes
                    # then fill it in with NaN
                    if mode_index < len(freqs):
                        freq_val = float(freqs[mode_index])
                        row[f"Mode{mode_index}_freq"] = float(freqs[mode_index])
                    else:
                        row[f"Mode{mode_index}_freq"] = np.nan

                    # eigenvector (length N) or NaN array
                    if mode_index < eigvecs.shape[0]:
                        omega_mode = 2.0 * pi * freq_val
                        scale = eigvec_scale * np.sqrt(1.0 / omega_mode)
                        row[f"Mode{mode_index}_eigvec"] = (scale * np.ravel(eigvecs[mode_index])).astype(float)
                    else:
                        # use full-length nan array to keep shape consistent
                        row[f"Mode{mode_index}_eigvec"] = np.full(N, np.nan, dtype=float)

                # --- Store completed row ---
                rows.append(row)

    return pd.DataFrame(rows)

def build_mode_series_and_combinations(df):
    """
    Inputs:
    df: dataframe constructed from tweezer_combos_full_radial

    Returns:
    dictionary of {"mode_series": mode_series, "mode_lists": mode_lists} where:
        mode_series = {i:df["Mode{i}_eigvec"] for i in mode_indices}
        mode_lists = {i: list of tuples (tweezed_ions, eigvec_array)}
    """

    # find Mode{i}_eigvec columns sorted by i
    # Mode 0 is always the center of mass mode
    mode_cols = sorted(
        [c for c in df.columns if re.match(r"^Mode\d+_eigvec$", c)],
        key=lambda c: int(re.match(r"Mode(\d+)_eigvec$", c).group(1)),
    )
    mode_indices = [int(re.match(r"Mode(\d+)_eigvec$", c).group(1)) for c in mode_cols]

    # makes a dictionary to map mode index to the original dataframe column
    #if you want to speed up the code, consider doing this in a different manner
    mode_series = {i: df[f"Mode{i}_eigvec"] for i in mode_indices}

    # Loop through all rows of df and get the ruples of (tweezed_ions, eigvec_array)
    # Results are stored in a dictionary
    mode_lists = {}
    for i in mode_indices:
        col = f"Mode{i}_eigvec"
        items = []
        for idx in df.index:
            # get tweezed-ion configuration, basically turn () into nan, and (i,) into i
            tweezed = df.at[idx, "Tweezed ions"]
            # replace empty tuple with np.nan
            if tweezed == ():
                tweezed = np.nan
            elif len(tweezed) == 1:
                tweezed = tweezed[0]  # if single-ion tuple, just use the integer
            val = df.at[idx, col]
            try:
                arr = np.asarray(val, dtype=float)
            except Exception:
                arr = np.atleast_1d(val)
            items.append((tweezed, arr))
        mode_lists[i] = items

    return {"mode_series": mode_series, "mode_lists": mode_lists}

def condense_by_min_abs(data):
    """
    input: output from "combine lists"

    output: [(tweezed ion),(ion-to-mode config),minimum absolute value of mode-participation from that config]
    """
    condensed = []
    for idx_group, combo, arr in data:
        # choose element with smallest abs() but keep real sign
        min_val = min(arr, key=lambda x: abs(x))
        condensed.append((idx_group, combo, min_val))
    return condensed

def filter_by_max_min_abs(data):
    """
    Input: output from condense by min abs

    Output: [(tweezed ion),(ion-to-mode config),maximum of all minimum absolute values of mode-participation]
                this output is one tuple for a tweezed ion 
    """
    if not data:
        return []
    
    #find the maximum value across each ion to mode config per a tweezed ion 
    max_abs = max(abs(t[2]) for t in data)
    return [t for t in data if abs(t[2]) == max_abs]

def combine_lists(*lists):
    """
    For use inside run optimal mode selection.
    Generates all index combinations across all ions and all modes
    only keeping combinations with unique indices (ie, no two modes can pick the same ion).
    Input:
        lists: a list of tuples [(tweezed_ion_config,eigenvector array)],one list per mode
    
    Outputs:
    A dictionary mapping each tweezed ion configuration to a list of tuples:
        {Tweezed Ion: 
            [Tweezed Ion,(ion-to-mode mapping), array of corresponding mode-coupling values]}
    """

    # Get the number of ions from the length of the input lists
    N = len(lists)
    # Get the number of modes from the length of the eigenvector arrays in the lists
    K = len(lists[0][0][1])

    # Collect all indices of tweezed ion 
    keys = [idx for idx, _ in lists[0]]

    # Converts list to dictionary because it apparently has a faster lookup time
    idx_maps = []
    for lst in lists:
        idx_maps.append({idx: arr for idx, arr in lst})
    groups = {key: [] for key in keys}

    # Loop over each tweezed ion configuration
    for key in keys:

        # gathering eig-vecs for this tweezed ion across all modes
        chosen = [idx_maps[m][key] for m in range(N)]

        # Creating all index combinations of every ion to every mode without repeating modes/ions
        for elem_choices in product(range(K), repeat=N):

            # Setting unique indices
            if len(set(elem_choices)) != N:
                continue

            # Build combined vector element-wise
            values = np.array([
                chosen[m][elem_choices[m]]
                for m in range(N)
            ])

            # Store tuple
            groups[key].append(
                (key, elem_choices, values)
            )

    return groups

def select_global_max_min_abs(groups, tol=1e-12):
    """
    Given a list of lists where each inner list contains tuples
    (idx_group, combo, value),
    return all tuples whose |value| equals the maximum absolute value
    across the entire dataset, allowing for floating-point tolerance.
    """

    # Flatten everything into one list of tuples
    all_tuples = [t for group in groups for t in group]

    if not all_tuples:
        return []

    # Compute global maximum |value|
    global_max = max(abs(t[2]) for t in all_tuples)

    # Collect all tuples that match this max within tolerance
    winners = [
        t for t in all_tuples
        if abs(abs(t[2]) - global_max) < tol
    ]

    return winners

def run_optimal_mode_selection_tweezed_only(
    omega_tweezer,
    linewidths,
    omega_res,
    m,
    mode_calc_r,
    N,
    f_rf_r,
    f_rf_a,
    P_opt,
    w0,
    max_tweezed=1
):
    """
    Inputs:
    omega_tweezer = optical tweezer beam angular frequency [2*Pi x Hz]
    linewidths = linewidth of the given resonant transition taken from NIST database in angular frequency units 
        (ex, S1/2 to P1/2 and S1/2 to P3/2 for 40Ca+) 
    omega_res = angular frequency of resonant transition, also based off NIST data [2*Pi x Hz]
    m = mass of ion [kg]
    mode_calc_r = function from above to calculate radial modes
    N_list = list of number of ions to loop over (can also be a single integer)
    f_rf_r = radial rf trapping frequency [Hz]
    f_rf_a = axial rf trapping frequency [Hz]
    P_opt = list of total optical power of tweezer laser beam to loop over (can also be a single number) [W]
    w0 = beamwaist of the tweezer laser beam [m] 
    
    Returns:
    A list of tuples
        [ (tweezed ion, (mode mapping), (corresponding mode-coupling contributions per ion)
                , P_opt) ]
    """
    
    # 1. Build Lamb-Dicke parameter lists for all configurations
    df = tweezer_combos_full_radial(
        omega_tweezer, linewidths, omega_res, m,
        mode_calc_r, N, f_rf_r, f_rf_a, P_opt, w0,
        max_tweezed=max_tweezed
    )
    
    result = build_mode_series_and_combinations(df)
    mode_lists_dict = result["mode_lists"]
    
  # Filter out P=0 rows (no actual tweezing)
    df = df[df['P_per_tweezer (W)'] > 1e-12]
    
    if df.empty:
        return []
    
    result = build_mode_series_and_combinations(df)
    mode_lists_dict = result["mode_lists"]
    
    # Get rid of all Nan and None keys that correspond to no tweezed ions
    cleaned = {
        k: v for k, v in mode_lists_dict.items()
        if k is not None and not (isinstance(k, float) and math.isnan(k)) and k != ()
    }
    mode_lists_dict = cleaned
    
    if not mode_lists_dict:
        return []
    
    # 2. Sort mode indices and collect all mode lists
    mode_indices = sorted(
        mode_lists_dict.keys(),
        key=lambda x: (str(x) if isinstance(x, tuple) else x)
    )
    mode_lists_ordered = [mode_lists_dict[i] for i in mode_indices]
    
    # 3. Combine modes across all tweezed ions
    combos = combine_lists(*mode_lists_ordered)
    
    # 4. Condense by min(abs)
    condensed = {k: condense_by_min_abs(v) for k, v in combos.items()}
    
    # 5. Filter by max(min(abs)) within each group
    best_each = {k: filter_by_max_min_abs(v) for k, v in condensed.items()}
    
    # 6. Select global winners
    winners = select_global_max_min_abs(list(best_each.values()))

    # 7. Compute best ion/mode indices properly
    augmented = []
    for tweezed_ion, mapping, score in winners:
        # Access the corresponding values from combos
        combo_dict = {combo_indices: arr for _, combo_indices, arr in combos[tweezed_ion]}
        full_lamb_dicke_vector = combo_dict.get(mapping, np.array(mapping))  
    
        #adding P_opt back to the tuple 
        augmented.append(
        ((tweezed_ion, mapping, score, P_opt, full_lamb_dicke_vector))  
        )

    return augmented

def run_optimal_mode_selection_untweezed_only(
    omega_tweezer,
    linewidths,
    omega_res,
    m,
    mode_calc_r,
    N,
    f_rf_r,
    f_rf_a,
    P_opt,
    w0
):
    """
    Inputs:
    omega_tweezer = optical tweezer beam angular frequency [2*Pi x Hz]
    linewidths = linewidth of the given resonant transition taken from NIST database in angular frequency units 
        (ex, S1/2 to P1/2 and S1/2 to P3/2 for 40Ca+) 
    omega_res = angular frequency of resonant transition, also based off NIST data [2*Pi x Hz]
    m = mass of ion [kg]
    mode_calc_r = function from above to calculate radial modes
    N_list = list of number of ions to loop over (can also be a single integer)
    f_rf_r = radial rf trapping frequency [Hz]
    f_rf_a = axial rf trapping frequency [Hz]
    P_opt = list of total optical power of tweezer laser beam to loop over (can also be a single number) [W]
    w0 = beamwaist of the tweezer laser beam [m] 
    
    Returns:
    A list of tuples
        [ (tweezed ion, (mode mapping), (corresponding mode-coupling contributions per ion)
                , P_opt) ]
    """
    
    # 1. Build Lamb-Dicke parameter lists for the untweezed configuration
    df = tweezer_combos_full_radial(
        omega_tweezer, linewidths, omega_res, m,
        mode_calc_r, N, f_rf_r, f_rf_a, P_opt, w0,
        max_tweezed=0
    )
    
    result = build_mode_series_and_combinations(df)
    mode_lists_dict = result["mode_lists"]
    
    # ---- CLEAN: remove unusable keys ----
    cleaned = {
        k: v for k, v in mode_lists_dict.items()
        if k is not None and not (isinstance(k, float) and math.isnan(k)) and k != ()
    }
    mode_lists_dict = cleaned
    
    if not mode_lists_dict:
        return []
    
    # 2. Sort mode indices numerically if possible
    mode_indices = sorted(
        mode_lists_dict.keys(),
        key=lambda x: (str(x) if isinstance(x, tuple) else x)
    )
    
    if not mode_indices:
        return []
    
    # 3. Collect lists in canonical order
    mode_lists_ordered = [mode_lists_dict[i] for i in mode_indices]
    
    # 4. Combine modes
    combos = combine_lists(*mode_lists_ordered)
    
    # 5. Condense by min(abs)
    condensed = {k: condense_by_min_abs(v) for k, v in combos.items()}
    
    # 6. Filter by max(min(abs)) within each group
    best_each = {k: filter_by_max_min_abs(v) for k, v in condensed.items()}
    
    # 7. Select global winners
    winners = select_global_max_min_abs(list(best_each.values()))
    
    # 8. Compute best ion/mode indices properly
    augmented = []
    for tweezed_ion, mapping, score in winners:
        # Access the corresponding values from combos
        combo_dict = {combo_indices: arr for _, combo_indices, arr in combos[tweezed_ion]}
        full_lamb_dicke_vector = combo_dict.get(mapping, np.array(mapping))
        
        # worst ion index = position in vector with largest magnitude
        worst_ion_index = int(np.argmax(np.abs(full_lamb_dicke_vector)))
        worst_mode_index = mapping[worst_ion_index]

        augmented.append(
            (
                (tweezed_ion, mapping, score, P_opt, full_lamb_dicke_vector)
            )
        )

    return augmented

def run_optimal_mode_selection_tweezed_power_sweep(
    omega_tweezer,
    linewidths,
    omega_res,
    m,
    mode_calc_r,
    N,
    f_rf_r,
    f_rf_a,
    P_list,
    w0,
    max_tweezed=1
):
    
    """
    Inputs:
    omega_tweezer = optical tweezer beam angular frequency [2*Pi x Hz]
    linewidths = linewidth of the given resonant transition taken from NIST database in angular frequency units 
        (ex, S1/2 to P1/2 and S1/2 to P3/2 for 40Ca+) 
    omega_res = angular frequency of resonant transition, also based off NIST data [2*Pi x Hz]
    m = mass of ion [kg]
    mode_calc_r = function from above to calculate radial modes
    N_list = list of number of ions to loop over (can also be a single integer)
    f_rf_r = radial rf trapping frequency [Hz]
    f_rf_a = axial rf trapping frequency [Hz]
    P_opt = list of total optical power of tweezer laser beam to loop over (can also be a single number) [W]
    w0 = beamwaist of the tweezer laser beam [m] 
    
    Returns:
    A list of tuples
        [ (tweezed ion, (mode mapping), (corresponding mode-coupling contributions per ion)
                , P_opt) ]
    """
    power_results = []

    for P_opt in P_list:
        winners = run_optimal_mode_selection_tweezed_only(
            omega_tweezer,
            linewidths,
            omega_res,
            m,
            mode_calc_r,
            N,
            f_rf_r,
            f_rf_a,
            P_opt,
            w0,
            max_tweezed=max_tweezed
        )
        power_results.append(winners)
    return power_results

def combine_lists_same_index_df(*lists):
    """
    N-dimensional version where all lists must pick
    the SAME component index for each vector.

    Returns a tidy pandas DataFrame.
    """

    # Get the number of modes from the length of the lists
    N = len(lists)

    # Get the number of ions from the length of the eig-vec 
    K = len(lists[0][0][1])

    # Collect one index at a time 
    keys = [idx for idx, _ in lists[0]]

    # Converts list to dictionary because it apparently has a faster lookup time
    idx_maps = []
    for lst in lists:
        idx_maps.append({idx: arr for idx, arr in lst})

    # Rows to accumulate for the DataFrame
    rows = []

    # Loop over each index group
    for key in keys:

        # Vectors for this index across all input lists
        chosen = [idx_maps[m][key] for m in range(N)]

        # Sweep SAME element index across all lists
        for elem in range(K):

            values = np.array([chosen[m][elem] for m in range(N)])

            rows.append({
                "Tweezed Ion": key,
                "Coolant Ion": elem,
            # "elem_indices_tuple": (elem,) * N,
                "Mode couplings": values
            })

    # Turn into DataFrame
    df = pd.DataFrame(rows)

    # add inverse and summed-inverse columns (as requested)
    df["inverse_mode_couplings"] = df["Mode couplings"].apply(
        lambda arr: np.array([1.0/abs(x) if x != 0 else np.inf for x in arr])
    )
    df["sum_inverse_middle"] = df["inverse_mode_couplings"].apply(
        lambda lst: sum(x for x in lst if pd.notna(x))
    )

    return df

def midcircuit_modes(omega_tweezer,
                     linewidths,
                     omega_res,
                     m,
                     mode_calc_r,
                     N,
                     f_rf_r,
                     f_rf_a,
                     P,
                     w0
                     ):
    """
    Inputs:
    omega_tweezer = optical tweezer beam angular frequency [2*Pi x Hz]
    linewidths = linewidth of the given resonant transition taken from NIST database in angular frequency units 
        (ex, S1/2 to P1/2 and S1/2 to P3/2 for 40Ca+) 
    omega_res = angular frequency of resonant transition, also based off NIST data [2*Pi x Hz]
    m = mass of ion [kg]
    mode_calc_r = function from above to calculate radial modes
    N_list = list of number of ions to loop over (can also be a single integer)
    f_rf_r = radial rf trapping frequency [Hz]
    f_rf_a = axial rf trapping frequency [Hz]
    P_opt = list of total optical power of tweezer laser beam to loop over (can also be a single number) [W]
    w0 = beamwaist of the tweezer laser beam [m] 
    
    Returns:
    dataframe with columns:
    - "Tweezed Ion": which ion is tweezed in this configuration (NaN/None if no tweezed ion) (int)
    - "Coolant Ion": which ion is the best coolant candidate for this configuration (int)
    - "Mode couplings": array of mode coupling values for the coolant ion in this configuration (numpy array)
    - "inverse_mode_couplings": array of 1/abs(mode coupling) (numpy array)
    - "sum_inverse_middle": sum of the inverse_mode_couplings for this row (float)
    """
    # get all mode strucures for all possible tweezed ion positions
    results = tweezer_combos_full_radial(
        omega_tweezer,
        linewidths,
        omega_res,
        m,
        mode_calc_r,
        N,
        f_rf_r,
        f_rf_a,
        P,
        w0,
        max_tweezed=1,
    )
    # build the mode lists
    result = build_mode_series_and_combinations(results)
    mode_list_test = []
    for i in range(len(result) + 1):
        mode_list_test.append(result["mode_lists"][i])

    # make a data frame for the all modes from one ion across every ion for every tweezer configuration
    middle = combine_lists_same_index_df(*mode_list_test)

    # Filter the dataframe to pick only the best ion to cool and the best ion to tweeze 
    best_rows = middle.iloc[0:0].copy()
    tweezed_keys = [k for k in middle["Tweezed Ion"].unique() if pd.notna(k)]

    for key in tweezed_keys:
        subset = middle[middle["Tweezed Ion"] == key]
        if subset.empty:
            continue
        
        # First, considering Coolant Ion == Tweezed Ion and finding the smallest sum_inverse_middle
        subset_same = subset[subset["Coolant Ion"] == key]
        if not subset_same.empty:
            mn_same = subset_same["sum_inverse_middle"].abs().min()
            mask_same = np.isclose(subset_same["sum_inverse_middle"].abs(), mn_same, rtol=1e-12, atol=1e-12)
            tied_same = subset_same.loc[mask_same]
            chosen_same = tied_same.sort_values("Coolant Ion", na_position="last").iloc[0:1]
            best_rows = pd.concat([best_rows, chosen_same], ignore_index=True)
        
        # Next, considering coolant ion =/= tweezed ion and finding the smallest sum_inverse_middle
        subset_diff = subset[subset["Coolant Ion"] != key]
        if not subset_diff.empty:
            mn_diff = subset_diff["sum_inverse_middle"].abs().min()
            mask_diff = np.isclose(subset_diff["sum_inverse_middle"].abs(), mn_diff, rtol=1e-12, atol=1e-12)
            tied_diff = subset_diff.loc[mask_diff]
            chosen_diff = tied_diff.sort_values("Coolant Ion", na_position="last").iloc[0:1]
            best_rows = pd.concat([best_rows, chosen_diff], ignore_index=True)

    # Pick the winner across both cases, lowest sum_inverse_middle
    if best_rows.empty:
        return best_rows

    global_min = best_rows["sum_inverse_middle"].abs().min()
    keep_mask = np.isclose(best_rows["sum_inverse_middle"].abs(), global_min, rtol=1e-8, atol=1e-12)
    final = best_rows.loc[keep_mask].reset_index(drop=True)

    return final

def midcircuit_modes_untweezed(omega_tweezer,
                     linewidths,
                     omega_res,
                     m,
                     mode_calc_r,
                     N,
                     f_rf_r,
                     f_rf_a,
                     P,
                     w0
                     ):
    """
    Inputs:
    omega_tweezer = optical tweezer beam angular frequency [2*Pi x Hz]
    linewidths = linewidth of the given resonant transition taken from NIST database in angular frequency units 
        (ex, S1/2 to P1/2 and S1/2 to P3/2 for 40Ca+) 
    omega_res = angular frequency of resonant transition, also based off NIST data [2*Pi x Hz]
    m = mass of ion [kg]
    mode_calc_r = function from above to calculate radial modes
    N_list = list of number of ions to loop over (can also be a single integer)
    f_rf_r = radial rf trapping frequency [Hz]
    f_rf_a = axial rf trapping frequency [Hz]
    P_opt = list of total optical power of tweezer laser beam to loop over (can also be a single number) [W]
    w0 = beamwaist of the tweezer laser beam [m] 
    
    Returns:
    dataframe with columns:
    - "Tweezed Ion": Always Nan or None here
    - "Coolant Ion": which ion is the best coolant candidate for this configuration (int)
    - "Mode couplings": array of mode coupling values for the coolant ion in this configuration (numpy array)
    - "inverse_mode_couplings": array of 1/abs(mode coupling) (numpy array)
    - "sum_inverse_middle": sum of the inverse_mode_couplings for this row (float)
    """
    results = tweezer_combos_full_radial(
        omega_tweezer,
        linewidths,
        omega_res,
        m,
        mode_calc_r,
        N,
        f_rf_r,
        f_rf_a,
        P,
        w0,
        max_tweezed=0,
    )
    result = build_mode_series_and_combinations(results)

    mode_list_test = []
    for i in range(len(result) + 1):
        mode_list_test.append(result["mode_lists"][i])

    # use untweezed combiner that does not expect a 'Tweezed Ion' column
    middle = combine_lists_same_index_df(*mode_list_test)

    if middle.empty:
        return middle

    # Require "Mode couplings" present
    if "Mode couplings" not in middle.columns:
        return middle

    # inverse_mode_couplings = 1 / abs(value) for each entry in Mode couplings
    def inv_abs_list(arr):
        # arr may be list-like or numpy array; try to handle mixed/invalid entries robustly
        try:
            a = np.array([float(x) for x in arr], dtype=float)
        except Exception:
            # fallback: iterate and coerce elementwise (preserve invalid as np.nan)
            out = []
            for x in arr:
                try:
                    out.append(float(x))
                except Exception:
                    out.append(np.nan)
            a = np.array(out, dtype=float)

        a = np.abs(a)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(np.isnan(a), np.nan, np.where(a == 0.0, np.inf, 1.0 / a))
        return inv.astype(float)

    # store the actual inverses of the absolute couplings
    middle["inverse_mode_couplings"] = middle["Mode couplings"].apply(inv_abs_list)

    # sum_inverse_middle = sum of those inverses (treat NaN as missing, include np.inf if present)
    middle["sum_inverse_middle"] = middle["inverse_mode_couplings"].apply(lambda arr: float(np.nansum(arr)))

    # select rows with minimal sum_inverse_middle (best candidate(s))
    global_min = middle["sum_inverse_middle"].abs().min()
    keep_mask = np.isclose(middle["sum_inverse_middle"].abs(), global_min, rtol=1e-8, atol=1e-12)
    final = middle.loc[keep_mask].reset_index(drop=True)
    return final

def collect_midcircuit_combined_df(
    Ns,
    omega_t,
    linewidths_,
    omega_res_,
    m_,
    mode_calc,
    f_rf_r_,
    f_rf_a_,
    P_,
    w0_,
    return_many_N=False,
):
    """
    Run midcircuit_modes for each N in Ns and return a combined DataFrame.
    Does NOT print or display per-N results. Keeps all rows (no per-N dedupe).
    If return_many_N is True, also return the list of (N, per-N DataFrame).
    """
    dfs = []
    many_N = []

    for N in Ns:
        test_loop = midcircuit_modes(
            omega_t,
            linewidths_,
            omega_res_,
            m_,
            mode_calc,
            N,
            f_rf_r_,
            f_rf_a_,
            P_,
            w0_,
        )
        many_N.append((N, test_loop))

        if isinstance(test_loop, pd.DataFrame) and not test_loop.empty:
            df = test_loop.copy().reset_index(drop=True)
            df["N"] = int(N)
            try:
                df["Tweezed Ion"] = df["Tweezed Ion"].astype(int)
            except Exception:
                pass
            dfs.append(df)

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True, sort=False)
        cols = [
            "N",
            "Tweezed Ion",
            "Coolant Ion",
            "Mode couplings",
            "inverse_mode_couplings",
            "sum_inverse_middle",
        ]
        combined_df = combined_df[[c for c in cols if c in combined_df.columns]]
    else:
        combined_df = pd.DataFrame()

    if return_many_N:
        return combined_df, many_N
    return combined_df