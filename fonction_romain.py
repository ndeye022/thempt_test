
# Liste des fonctions nécessaires
# Romain Privat

# Bibliothèques Python : 
import numpy as np
import matplotlib.pyplot as plt
import sys
import numpy.polynomial.polynomial as nppol # résolution équation polynomiale

# Autres constantes :
Rgaz = 8.314472 # Constante des gaz parfaits, J/mol/K
onethird = 1./3. # puissance 1/3
eps = 1e-14 # tolérance
itmax = int(1e5)

# Constantes de l'équation de Van der Waals : 
r1 = -1. - np.sqrt(2.)
r2 = -1. + np.sqrt(2.)
sumr = r1 + r2

compac_crit = 1./(((1-r1)*(1-r2)**2)**onethird + ((1-r2)*(1-r1)**2)**onethird + 1.) # compacité critique = b/Vcrit
zcrit = 1. / (3.-compac_crit*(1+sumr)) # facteur de compressibilité critique
omega_b = compac_crit * zcrit
omega_a = (1-compac_crit*r1)*(1-compac_crit*r2)*(2-compac_crit*sumr) /((1-compac_crit)*(3-compac_crit*(1+sumr))**2)

# Fonction Calcul_b_acrit_zcrit()) : 
#==============================================================================
def Calcul_b_acrit_zcrit_vcrit(Tc,Pc):
    # Outputs : covolume b (m3/mol), acrit = a(Tc), 
    #           Facteur compress. crit, zcrit, 
    #           volume molaire critique vcrit, m3/mol
    RTc = Rgaz*Tc
    acrit = omega_a*RTc**2 / Pc # a(Tc)
    b = omega_b * RTc / Pc # covolume, m3/mol
    vcrit = zcrit * RTc / Pc # Volume molaire critique, m3/mol
    return b, acrit, zcrit, vcrit

# Fonction a(T) : 
#==============================================================================
def m(w): # facteur de forme
    return 0.37464 + 1.54226*w - 0.26992*w**2 # m de PR

def Coeff_a_EoS(T,Tc,acrit,w,ider): # paramètre attractif a, Pa.m^6 / mol²
    # Inputs = temp. T (K), temp. crit. Tc (K), fact. acent. w, ider
    # ider : niveau de dérivation souhaité
    #    ider = 0, a
    #    ider = 1, a'(T)
    #    ider = 2, a''(T)
    fact_m = m(w)
    foncT = 1+fact_m*(1. - np.sqrt(T/Tc))
    a_EoS = acrit*foncT**2
    da_EoS = 0. 
    d2a_EoS = 0. 
    
    if ider >= 1:
        da_EoS = -acrit*fact_m*foncT / np.sqrt(T*Tc) # da / dT
    if ider >= 2:
        d2a_EoS = 0.5*acrit*fact_m*(fact_m+1)/np.sqrt(Tc)*T**(-1.5) # d²a / dT²
    return a_EoS, da_EoS, d2a_EoS

# Fonction P(T,v,a,b,c), pression en Pa
# Inputs : T, temperature (K) et v, vol. mol en m3/mol
# Inputs : a, param attractif (Pa.m^6/mol²), b, covolume (m3/mol), 
#          c, correction volumique (m3/mol)
#==============================================================================
def P_EoS(T,v,a,b,c): # Pa
    v_nt = v + c # volume non translaté
    return Rgaz*T/(v_nt-b) - a/(v_nt-r1*b)/(v_nt-r2*b) 

# log nat. de la fugacité
# Inputs : T, temperature (K) et v, vol. mol en m3/mol
# Inputs : a, param attractif (Pa.m^6/mol²), b, covolume (m3/mol), 
#          c, correction volumique (m3/mol)
#==============================================================================
def lnfug_EoS(T,v,a,b,c):
    RT = Rgaz*T
    pres = P_EoS(T,v,a,b,c)
    v_nt = v + c # volume non translaté
    return pres*v_nt/RT - 1. - np.log((v_nt-b)/RT) \
        + a/(RT*b*(r1-r2))*np.log((v_nt-b*r1)/(v_nt-b*r2)) \

# Fonction hres_EoS(T,v,a,b,c), pression en Pa
# Inputs : T, temperature (K) et v, vol. mol en m3/mol
# Inputs : a, param attractif (Pa.m^6/mol²), b, covolume (m3/mol), 
#          c, correction volumique (m3/mol)
#==============================================================================
def hres_EoS(T,v,a,b,c,da): # Pa
    RT = Rgaz*T
    pres = P_EoS(T,v,a,b,c)
    v_nt = v + c # volume non translaté
    vbr1 = v_nt-b*r1
    vbr2 = v_nt-b*r2
    return RT*b/(v_nt-b) - a*v_nt/(vbr1*vbr2) \
        + 1/(b*(r1-r2))*(a - T*da)*np.log(vbr1/vbr2) \

# Fonction cPres_EoS(T,v,a,b,c), pression en Pa
# Inputs : T, temperature (K) et v, vol. mol en m3/mol
# Inputs : a, param attractif (Pa.m^6/mol²), b, covolume (m3/mol), 
#          c, correction volumique (m3/mol)
#==============================================================================
def Cpres_EoS(T,v,a,b,c,da,d2a): # Pa
    RT = Rgaz*T
    v_nt = v + c # volume non translaté
    vb = v_nt - b
    vbr1 = v_nt-b*r1
    vbr2 = v_nt-b*r2
    Cvres = -T/(b*(r1-r2))*d2a*np.log(vbr1/vbr2) 
    kTv = 1/(RT/vb**2 - a*(2*v_nt- (r1+r2)*b)/(vbr1*vbr2)**2)
    betaP = Rgaz/(v_nt-b) - da/(vbr1*vbr2)
    return Cvres - Rgaz + T * kTv * betaP**2

# Fonction cPGP(T), pression en Pa
# Equation 127 de la DIPPR
#==============================================================================
def CpGP(CoeffCpGP,T): # J/mol/K
    A, B, C, D, E, F, G = CoeffCpGP
    term1 = A
    term2 = (B * C**2 / T**2) * (np.exp(C / T) / (np.exp(C / T) - 1)**2)
    term3 = (D * E**2 / T**2) * (np.exp(E / T) / (np.exp(E / T) - 1)**2)
    term4 = (F * G**2 / T**2) * (np.exp(G / T) / (np.exp(G / T) - 1)**2)
    return 1e-3*(term1 + term2 + term3 + term4) # J/mol/K

# Résolution de l'équation d'état à T et P fixées : 
# il s'agit de déterminer les racines v de l'EoS qui sont solutions
# d'une équation cubique polynomiale de degré 3.
# Inputs : T, temperature (K) et P, pression en Pa
# Inputs : a, param attractif (Pa.m^6/mol²), b, covolume (m3/mol), 
#          c, correction volumique (m3/mol)
# Output : vecteur des volumes molaires solutions de l'EoS
#==============================================================================

# -------------------
# Aparté PYTHON : résolution d'une équation polynomiale à partir de polyroots.
# Exemple : pour calculer les racines du polynôme x^2 + 3.x + 5 = 0, on écrit : 
#           X = nppol.polyroots([5, 3, 1])
# -------------------

# écriture généralisée d'une cubique : P = RT/(v-b) - a / [(v-b.r1)(v-b.r2)]
def resolEoS(T,P,a,b,c): # résolution d'une équation cubique à T (K) et P (Pa) fixées
    # Méthode : on résout l'EoS non translatée puis on corrige les volumes
    # obtenus en soustrayant c
    RT = Rgaz*T 
    coef3 = 1.
    coef2 = -(b*(r1 + r2 + 1) + RT/P)
    coef1 = b**2 * (r1*r2 + r1 + r2) + RT*b/P*(r1+r2) + a/P
    coef0 = -b*(r1*r2*b**2 + r1*r2*b*RT/P + a/P)
    volumes = nppol.polyroots([coef0, coef1, coef2, coef3])
    
    # Sélection des racines réelles :
    volumes = volumes.real[abs(volumes.imag)<1e-30]

    # suppression des racines non physiques telles que v < b:
    volumes = volumes[volumes >= b] - c
    return volumes

# Calcul d'une pression de vapeur saturante à T fixée : 
#==============================================================================
def Psat(T, a, b, c):
    # Psat en Pa,
    # Inputs : T en K, a, b et c (correction volumique) en m3/mol
    # Outputs : Psat, vliq, vgaz (vliq et vgaz = vol. mol du liq et du gaz 
    #                             en équilibre LV, m3/mol)
    
    if T<Tc:
        # étape 1 : recherche de Pmin et Pmax sur l'isotherme
        #=====================================================
        bnew = b - c
        stepv = 1e-2*bnew 

        # Recherche de Pmin:
        # On parcourt la branche décroissante de l'isotherme P(v)
        vnew = bnew # valeur initiale de v
        Pold = 1e300 # valeur initiale de Pold, arbitrairement affectée à +infini
        dP = -1. # valeur arbitraire
        
        it = 0 # compteur d'itérations
        while dP<0: # tant que P diminue
            vold = vnew
            vnew = vold + stepv
            Pnew = P_EoS(T,vnew,a,b,c)
            dP = Pnew - Pold
            Pold = Pnew
            it += 1
            if it > itmax:
                print("Trop d'itérations - STOP")
                sys. exit()

        Pmin = Pold
        vmin = vold
        if Pmin < 0.:
            Pmin = 0.

        # Recherche de Pmax:
        # On parcourt la branche croissante de l'isotherme P(v)
        it = 0 # compteur d'itérations
        while dP>0: # tant que P augmente
            vold = vnew
            vnew = vold + stepv
            Pnew = P_EoS(T,vnew,a,b,c)
            dP = Pnew - Pold
            Pold = Pnew
            it += 1
            if it > itmax:
                print("Trop d'itérations - STOP")
                sys. exit()

        Pmax = Pold
        vmax = vold

        # étape 2 : dichotomie
        #======================
        Nb_iter = int(np.floor(np.log((Pmax-Pmin)/eps)/np.log(2.)) + 1)
        
        for i in range(Nb_iter):
            P0 = 0.5*(Pmin + Pmax)
            VolMol = resolEoS(T,P0,a,b,c)
            volLiq = min(VolMol)
            volGaz = max(VolMol)
            
            # calcul de delta = ln f_liq - ln f_Gaz
            delta = lnfug_EoS(T,volLiq,a,b,c) - lnfug_EoS(T,volGaz,a,b,c)
            if delta > 0:
                Pmin = P0
            else:
                Pmax = P0
        Psat = P0
        vliq = volLiq
        vgaz = volGaz       
    else:
        Psat = 0.
        vliq = 0.
        vgaz = 0.
    return Psat, vliq, vgaz

# Calcul du paramètre de translation volumique c de l'EoS : 
#==============================================================================
def Calcul_c(vliq08, Tc, acrit, w, b):
    # Outputs : - cvol, correction volumique en m3/mol
    #             Elle est telle que : vEoS = vEoS non corrigée - cvol
    #           - bnew, m3/mol, nouveau covolume = b EoS non corrigée - cvol
    c = 0
    T08 = 0.8*Tc
    # Calcul du param attractif a à la température T :
    a, da, d2a = Coeff_a_EoS(T08,Tc,acrit,w,0) # ider = 0

    # Calcul du vol mol liq saturé à Tr = 0.8
    bidon, veos_liq08, bidon = Psat(T08, a, b, c)
    # Calcul de la translation vol :
    # c = Vliq EoS originale (Tr = 0.8) - Vliq exp (Tr = 0.8)
    cvol = veos_liq08 - vliq08
    bnew = b - cvol
    return cvol, bnew

# Pilote de calcul de Psat : 
#==============================================================================
def Pilote_calcul_Psat(T, Tc, Pc, w, vliq08):
    # Calcul de b et acrit :
    b, acrit, zcrit, vcrit = Calcul_b_acrit_zcrit_vcrit(Tc, Pc)

    # Calcul de c :
    c, bnew = Calcul_c(vliq08, Tc, acrit, w, b)

    if np.isscalar(T):
        # Calcul du param attractif a à la température T :
        a, da, d2a = Coeff_a_EoS(T,Tc,acrit,w,0)
        Ps, vliq, vgaz = Psat(T, a, b, c)
    else:
        Ps = [] 
        vliq = [] 
        vgaz = []
        for i in range(len(T)):
            a, da, d2a = Coeff_a_EoS(T[i],Tc,acrit,w,0)
            ValPs, ValVl, ValVg = Psat(T[i], a, b, c)
            Ps = np.append(Ps,ValPs)
            vliq = np.append(vliq,ValVl) 
            vgaz = np.append(vgaz,ValVg)
    return Ps, vliq, vgaz 
