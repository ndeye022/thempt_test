#!/usr/bin/env python
# coding: utf-8
# Fusion : Interface moderne (code 2) + logique EOS (code 1)
# LRGP Nancy - 2025

import os
import sys
import warnings
import numpy as np
import numpy.polynomial.polynomial as nppol
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import streamlit as st
import rdkit
import rdkit.Chem.inchi

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from mordred import Calculator, descriptors as mordred_descriptors
from sklearn.base import BaseEstimator, RegressorMixin
from streamlit_ketcher import st_ketcher

warnings.simplefilter("ignore")

######################################################################################################
## Configuration de la page
######################################################################################################

st.set_page_config(
    page_title="Prédiction des propriétés de composés chimiques - Plateforme IA - LRGP - Nancy",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1F4E79;
        margin-bottom: 0rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .result-card {
        background-color: #EBF3FB;
        border-left: 5px solid #2E75B6;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FFF3CD;
        border-left: 5px solid #FFC107;
        padding: 0.8rem;
        border-radius: 6px;
    }
    .footer {
        font-size: 0.8rem;
        color: #999;
        text-align: center;
        margin-top: 3rem;
    }
    </style>
""", unsafe_allow_html=True)

######################################################################################################
## Fonctions EOS de Romain Privat
######################################################################################################

Rgaz = 8.314472
onethird = 1./3.
eps = 1e-14
itmax = int(1e5)

r1 = -1. - np.sqrt(2.)
r2 = -1. + np.sqrt(2.)
sumr = r1 + r2

compac_crit = 1./(((1-r1)*(1-r2)**2)**onethird + ((1-r2)*(1-r1)**2)**onethird + 1.)
zcrit = 1. / (3.-compac_crit*(1+sumr))
omega_b = compac_crit * zcrit
omega_a = (1-compac_crit*r1)*(1-compac_crit*r2)*(2-compac_crit*sumr) / ((1-compac_crit)*(3-compac_crit*(1+sumr))**2)

CoeffCpGP = [33257.8886, 70253.576162, -1605.767453, 60622.500697, -3784.682119, 12107.569262, 489.57675]


def Calcul_b_acrit_zcrit_vcrit(Tc, Pc):
    RTc = Rgaz * Tc
    acrit = omega_a * RTc**2 / Pc
    b = omega_b * RTc / Pc
    vcrit = zcrit * RTc / Pc
    return b, acrit, zcrit, vcrit


def m(w):
    return 0.37464 + 1.54226*w - 0.26992*w**2


def Coeff_a_EoS(T, Tc, acrit, w, ider):
    fact_m = m(w)
    foncT = 1 + fact_m*(1. - np.sqrt(T/Tc))
    a_EoS = acrit * foncT**2
    da_EoS = 0.
    d2a_EoS = 0.
    if ider >= 1:
        da_EoS = -acrit * fact_m * foncT / np.sqrt(T*Tc)
    if ider >= 2:
        d2a_EoS = 0.5*acrit*fact_m*(fact_m+1)/np.sqrt(Tc)*T**(-1.5)
    return a_EoS, da_EoS, d2a_EoS


def P_EoS(T, v, a, b, c):
    v_nt = v + c
    return Rgaz*T/(v_nt-b) - a/(v_nt-r1*b)/(v_nt-r2*b)


def lnfug_EoS(T, v, a, b, c):
    RT = Rgaz*T
    pres = P_EoS(T, v, a, b, c)
    v_nt = v + c
    return pres*v_nt/RT - 1. - np.log((v_nt-b)/RT) \
        + a/(RT*b*(r1-r2))*np.log((v_nt-b*r1)/(v_nt-b*r2))


def hres_EoS(T, v, a, b, c, da):
    RT = Rgaz*T
    pres = P_EoS(T, v, a, b, c)
    v_nt = v + c
    vbr1 = v_nt - b*r1
    vbr2 = v_nt - b*r2
    return RT*b/(v_nt-b) - a*v_nt/(vbr1*vbr2) \
        + 1/(b*(r1-r2))*(a - T*da)*np.log(vbr1/vbr2)


def Cpres_EoS(T, v, a, b, c, da, d2a):
    RT = Rgaz*T
    v_nt = v + c
    vb = v_nt - b
    vbr1 = v_nt - b*r1
    vbr2 = v_nt - b*r2
    Cvres = -T/(b*(r1-r2))*d2a*np.log(vbr1/vbr2)
    kTv = 1/(RT/vb**2 - a*(2*v_nt - (r1+r2)*b)/(vbr1*vbr2)**2)
    betaP = Rgaz/(v_nt-b) - da/(vbr1*vbr2)
    return Cvres - Rgaz + T * kTv * betaP**2


def CpGP(CoeffCpGP, T):
    A, B, C, D, E, F, G = CoeffCpGP
    term1 = A
    term2 = (B * C**2 / T**2) * (np.exp(C/T) / (np.exp(C/T) - 1)**2)
    term3 = (D * E**2 / T**2) * (np.exp(E/T) / (np.exp(E/T) - 1)**2)
    term4 = (F * G**2 / T**2) * (np.exp(G/T) / (np.exp(G/T) - 1)**2)
    return 1e-3*(term1 + term2 + term3 + term4)


def resolEoS(T, P, a, b, c):
    RT = Rgaz*T
    coef3 = 1.
    coef2 = -(b*(r1 + r2 + 1) + RT/P)
    coef1 = b**2 * (r1*r2 + r1 + r2) + RT*b/P*(r1+r2) + a/P
    coef0 = -b*(r1*r2*b**2 + r1*r2*b*RT/P + a/P)
    volumes = nppol.polyroots([coef0, coef1, coef2, coef3])
    volumes = volumes.real[abs(volumes.imag) < 1e-30]
    volumes = volumes[volumes >= b] - c
    return volumes


def Psat_F(T, a, b, c, Tc, Pc):
    if T < Tc:
        bnew = b - c
        stepv = 1e-2*bnew
        vnew = bnew
        Pold = 1e300
        dP = -1.
        it = 0
        while dP < 0:
            vold = vnew
            vnew = vold + stepv
            Pnew = P_EoS(T, vnew, a, b, c)
            dP = Pnew - Pold
            Pold = Pnew
            it += 1
            if it > itmax:
                break
        Pmin = Pold
        if Pmin < 0.:
            Pmin = 0.
        it = 0
        while dP > 0:
            vold = vnew
            vnew = vold + stepv
            Pnew = P_EoS(T, vnew, a, b, c)
            dP = Pnew - Pold
            Pold = Pnew
            it += 1
            if it > itmax:
                break
        Pmax = Pold
        Nb_iter = int(np.floor(np.log((Pmax-Pmin)/eps)/np.log(2.)) + 1)
        for i in range(Nb_iter):
            P0 = 0.5*(Pmin + Pmax)
            VolMol = resolEoS(T, P0, a, b, c)
            volLiq = min(VolMol)
            volGaz = max(VolMol)
            delta = lnfug_EoS(T, volLiq, a, b, c) - lnfug_EoS(T, volGaz, a, b, c)
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


def Calcul_c(vliq08, Tc, acrit, w, b, Pc):
    c = 0
    T08 = 0.8*Tc
    a, da, d2a = Coeff_a_EoS(T08, Tc, acrit, w, 0)
    bidon, veos_liq08, bidon2 = Psat_F(T08, a, b, c, Tc, Pc)
    cvol = veos_liq08 - vliq08
    bnew = b - cvol
    return cvol, bnew


def Pilote_calcul_Psat(T, Tc, Pc, w, vliq08):
    b, acrit, zcrit_val, vcrit = Calcul_b_acrit_zcrit_vcrit(Tc, Pc)
    c, bnew = Calcul_c(vliq08, Tc, acrit, w, b, Pc)
    if np.isscalar(T):
        a, da, d2a = Coeff_a_EoS(T, Tc, acrit, w, 0)
        Ps, vliq, vgaz = Psat_F(T, a, b, c, Tc, Pc)
    else:
        Ps = []
        vliq = []
        vgaz = []
        for i in range(len(T)):
            a, da, d2a = Coeff_a_EoS(T[i], Tc, acrit, w, 0)
            ValPs, ValVl, ValVg = Psat_F(T[i], a, b, c, Tc, Pc)
            Ps = np.append(Ps, ValPs)
            vliq = np.append(vliq, ValVl)
            vgaz = np.append(vgaz, ValVg)
    return Ps, vliq, vgaz


######################################################################################################
## Classe MetaModel et chargement des modèles IA
######################################################################################################

class MetaModel(BaseEstimator, RegressorMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        return sum(predictions) / len(self.models)


@st.cache_resource(show_spinner="Chargement des modèles IA...")
def load_models():
    base = os.path.dirname(os.path.abspath(__file__))
    TC   = joblib.load(os.path.join(base, '01_modele_final_TC.joblib'))
    PC   = joblib.load(os.path.join(base, '02_modele_final_PC.joblib'))
    ACEN = joblib.load(os.path.join(base, '03_modele_final_ACEN.joblib'))
    NBP  = joblib.load(os.path.join(base, '04_modele_final_NBP.joblib'))
    TTR  = joblib.load(os.path.join(base, '05_modele_final_TTR.joblib'))
    VLIQ = joblib.load(os.path.join(base, '08_modele_final_VLIQ.joblib'))
    return TC, PC, ACEN, NBP, TTR, VLIQ


@st.cache_data(show_spinner=False)
def charger_colonnes():
    base = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base, 'noms_colonnes_247_TC.txt'), 'r') as f:
        cols = [ligne.strip() for ligne in f]
    del cols[0]
    return cols


######################################################################################################
## Fonctions descripteurs et prédiction
######################################################################################################

symboles_elements = [
    "He", "Li", "Be", "Ne", "Na", "Mg", "Al", "Si", "Ar", "K", "Ca", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    "Zn", "Ga", "Ge", "As", "Se", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa",
    "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"
]


def verifier_symboles(chaine):
    for symbole in symboles_elements:
        if symbole in chaine:
            return True, symbole
    return False, None


def calculer_descripteurs_mordred(smiles_list):
    calc = Calculator(mordred_descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    return calc.pandas(mols)


def nettoyer_descripteurs(df):
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df.fillna(0).astype(float)


def valider_smiles(smiles):
    if not smiles or not smiles.strip():
        return False, None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return mol is not None, mol
    except Exception:
        return False, None


def predire_proprietes(smiles):
    smiles = smiles.strip()
    TC_model, PC_model, ACEN_model, NBP_model, TTR_model, VLIQ_model = load_models()
    noms_colonnes = charger_colonnes()
    df_desc = calculer_descripteurs_mordred([smiles])
    X = df_desc[noms_colonnes]
    X = nettoyer_descripteurs(X)
    tc   = TC_model.predict(X)[0]
    pc   = PC_model.predict(X)[0]
    acen = ACEN_model.predict(X)[0]
    nbp  = NBP_model.predict(X)[0]
    ttr  = TTR_model.predict(X)[0]
    vliq = VLIQ_model.predict(X)[0]

    # Corrections pour cas particuliers
    canon = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    if canon == 'C':
        pc = 46.030; nbp = 112.261; acen = 0.0120179982127699; ttr = 95.2282376595891
    if canon == 'O':
        tc = 645.4039572736; pc = 220.030

    atome_interdit, _ = verifier_symboles(smiles)
    if atome_interdit:
        return 88888, 88888, 88888, 88888, 88888, 88888

    return tc, pc, acen, nbp, ttr, vliq


######################################################################################################
## Fonction tracé EOS
######################################################################################################

def tracer_courbes_eos(tc, pc_bar, acen, vliq_cm3):
    """Calcule et trace les 4 graphiques EOS pour une molécule donnée."""
    Pc = pc_bar * 1e5  # Pa
    vliq08 = vliq_cm3 * 1e-6  # m3/mol

    b, acrit, zcrit_val, vcrit = Calcul_b_acrit_zcrit_vcrit(tc, Pc)
    c, bnew = Calcul_c(vliq08, tc, acrit, acen, b, Pc)

    Tmin = 0.75 * tc
    Tmax = np.floor(tc)
    DT = 0.5
    T_List = np.arange(Tmin, Tmax, DT)
    Psat_List = np.zeros(len(T_List))
    Vliq_List = np.zeros(len(T_List))
    Vgaz_List = np.zeros(len(T_List))

    res = Pilote_calcul_Psat(T_List, tc, Pc, acen, vliq08)
    Psat_List[:] = res[0] * 1e-5   # bar
    Vliq_List[:] = res[1] * 1e6    # mL/mol
    Vgaz_List[:] = res[2] * 1e6    # mL/mol

    # Point critique
    T_List    = np.append(T_List, tc)
    Psat_List = np.append(Psat_List, Pc*1e-5)
    Vliq_List = np.append(Vliq_List, vcrit*1e6)
    Vgaz_List = np.append(Vgaz_List, vcrit*1e6)

    vcrit_cm3  = vcrit * 1e6
    vliq08_cm3 = vliq08 * 1e6

    # ---- Graphe 1 : Courbe de pression de vapeur (P, T) ----
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(T_List, Psat_List, 'k')
    ax1.plot(tc, Pc*1e-5, 'ko')
    ax1.set_xlabel('$T$ (K)', fontsize=10)
    ax1.set_ylabel('$P^{sat}$ (bar)', fontsize=10)
    ax1.set_title("Vapor pressure curve $(P,T)$", fontsize=14)
    fig1.tight_layout()

    # ---- Graphe 2 : Courbe d'équilibre (T, densité) ----
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(1e3/Vliq_List, T_List, 'b', label="Branche liq.")
    ax2.plot(1e3/Vgaz_List, T_List, 'r', label="Branche gaz.")
    ax2.plot(1e3/vcrit_cm3, tc, 'ko', label="Pt. crit.")
    ax2.plot(1e3/vliq08_cm3, 0.8*tc, 'g*', label=r"$\rho_{liq}(T_r=0.8)$")
    ax2.set_xlabel(r'$\rho$ (mol/L)', fontsize=10)
    ax2.set_ylabel('$T$ (K)', fontsize=10)
    ax2.set_title(r"Vapor pressure curve $(T,\rho)$", fontsize=14)
    ax2.legend(fontsize=9)
    fig2.tight_layout()

    # ---- Graphe 3 : Enthalpie de vaporisation ----
    DvapH_List = np.zeros(len(T_List))
    CpLiq_List = []
    T_ListCp   = []

    for i in range(len(T_List)):
        T = T_List[i]
        a, da, d2a = Coeff_a_EoS(T, tc, acrit, acen, 2)
        vl = Vliq_List[i] * 1e-6
        vg = Vgaz_List[i] * 1e-6
        DvapH_List[i] = hres_EoS(T, vg, a, b, c, da) - hres_EoS(T, vl, a, b, c, da)
        if T < 0.98*tc:
            T_ListCp = np.append(T_ListCp, T)
            CpL = Cpres_EoS(T, vl, a, b, c, da, d2a) + CpGP(CoeffCpGP, T)
            CpLiq_List = np.append(CpLiq_List, CpL)

    DvapH_List[-1] = 0.

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.plot(T_List, DvapH_List*1e-3, 'b')
    ax3.set_xlabel(r'$T$ (K)', fontsize=10)
    ax3.set_ylabel(r'$\Delta_{vap}H$ (kJ/mol)', fontsize=10)
    ax3.set_title(r"Plan $(T, \Delta_{vap}H)$", fontsize=14)
    fig3.tight_layout()

    # ---- Graphe 4 : Cp liquide ----
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    ax4.plot(T_ListCp, CpLiq_List, 'b', label="Branche liq.")
    ax4.set_xlabel(r'$T$ (K)', fontsize=10)
    ax4.set_ylabel(r'$c_{P,liq}$ (J/mol/K)', fontsize=10)
    ax4.set_title(r"Plan $(c_{P,liq}, T)$", fontsize=14)
    ax4.legend(fontsize=9)
    fig4.tight_layout()

    return fig1, fig2, fig3, fig4


######################################################################################################
## Barre latérale
######################################################################################################

with st.sidebar:
    try:
        st.image("https://media.licdn.com/dms/image/v2/D4E0BAQETe5Myk-nlsQ/company-logo_200_200/company-logo_200_200/0/1727789760495/lrgp_nancy_logo?e=2147483647&v=beta&t=ch_7NJa6n6em_OwmOgdOrWuyfe5pSrvTkdimndUDzKk", width=180)
    except:
        pass
    st.markdown("---")
    st.markdown("### 🔬 À propos")
    st.info(
        "In this paper, we propose a robust deep-learning model based on a Quantitative Structure − Property Relationship" 
"(QSPR) approach for estimating the critical temperature (TC), critical pressure (PC), acentric factor (ACEN) and nor"
"mal boiling point (NBP) of any C, H, O, N, S, P, F, Cl, Br, I molecule. The Mordred calculator was used to determine "
"247 descriptors to characterize the molecules considered in this work. For each evaluated property, multiple neural "
"networks were trained within a bagging framework. The predictions from the final ensemble were successfully tested "
"against a large set of experimental data comprising more than 1700 molecules and compared with those from dif"
"ferent recent learning models found in the literature. Comprehensive comparisons and extensive testing highlight" 
"the robustness and predictive power of the newly proposed multimodal learning model."
    )
    st.markdown("---")
    st.markdown("### 📋 Propriétés prédites")
    st.markdown("""
- **TC** — Température critique (K)
- **PC** — Pression critique (bar)
- **ACEN** — Facteur acentrique (-)
- **NBP** — Point d'ébullition normal (K)
- **TTP** — Point triple (K)
- **VLIQ** — Volume liq. à Tr=0.8 (cm³/mol)
    """)
    st.markdown("---")
    st.markdown("### 💡 Exemples de SMILES")
    st.code("CCCC        → Butane")
    st.code("CCCCCC      → Hexane")
    st.code("c1ccccc1    → Benzène")
    st.code("CCO         → Éthanol")
    st.markdown("---")
    st.markdown(
        "<div class='footer'>LRGP — UMR CNRS 7274<br>ENSIC Nancy — 2025</div>",
        unsafe_allow_html=True
    )

######################################################################################################
## En-tête principal
######################################################################################################

# ── Image bannière en haut de page ──────────────────────────────────────────
st.image("B2.png", use_container_width=True)

st.markdown("<div class='main-title'>🔬 Plateforme IA — Propriétés Thermodynamiques</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>LRGP Nancy — Laboratoire Réactions et Génie des Procédés (UMR CNRS 7274)<br>"
            "<em>Roda Bounaceur, Francisco Paes, Romain Privat, Jean-Noël Jaubert</em></div>", unsafe_allow_html=True)
st.markdown("Pour plus d'informations : [Télécharger l'article (PDF)](https://rdcu.be/eC77w)")
st.markdown("---")

######################################################################################################
## Onglets principaux
######################################################################################################

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Prédiction via SMILES",
    "✏️ Dessiner une molécule",
    "📊 Prédiction par fichier",
    "⚗️ EOS — Équation d'état"
])

# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 1 : Prédiction via SMILES
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Prédiction à partir d'une notation SMILES")
    st.write("Entrez la notation SMILES de votre molécule pour obtenir ses propriétés thermodynamiques.")

    # Initialiser la clé session_state qui pilote le text_input
    if "smiles_tab1_val" not in st.session_state:
        st.session_state["smiles_tab1_val"] = ""

    # Boutons exemples : écrivent directement dans la clé du widget
    col_btn_ex = st.columns([1, 1, 1, 1])
    with col_btn_ex[0]:
        if st.button("💡 Hexane", use_container_width=True):
            st.session_state["smiles_tab1_val"] = "CCCCCC"
    with col_btn_ex[1]:
        if st.button("💡 Éthanol", use_container_width=True):
            st.session_state["smiles_tab1_val"] = "CCO"
    with col_btn_ex[2]:
        if st.button("💡 Benzène", use_container_width=True):
            st.session_state["smiles_tab1_val"] = "c1ccccc1"
    with col_btn_ex[3]:
        if st.button("💡 Butane", use_container_width=True):
            st.session_state["smiles_tab1_val"] = "CCCC"

    smiles_input = st.text_input(
        "Notation SMILES",
        key="smiles_tab1_val",
        placeholder="Ex: CCCC (butane), c1ccccc1 (benzène), CCO (éthanol)",
        help="La notation SMILES est une représentation textuelle de la structure d'une molécule."
    )

    predire = st.button("🔍 Prédire les propriétés", type="primary")

    smiles_clean = smiles_input.strip() if smiles_input else ""

    if predire and smiles_clean:
        valide, mol = valider_smiles(smiles_clean)
        if not valide:
            st.error("❌ SMILES invalide. Vérifiez la notation et réessayez.")
        else:
            atome_interdit, sym = verifier_symboles(smiles_clean)
            if atome_interdit:
                st.warning(f"⚠️ Atome non supporté détecté : `{sym}`")

            with st.spinner("⏳ Calcul des descripteurs et prédiction en cours..."):
                try:
                    tc, pc, acen, nbp, ttr, vliq = predire_proprietes(smiles_clean)
                    st.success("✅ Prédiction réussie !")
                    st.markdown("---")

                    col_mol, col_res = st.columns([1, 2])

                    with col_mol:
                        st.markdown("#### Structure moléculaire")
                        img = Draw.MolToImage(mol, size=(300, 250))
                        st.image(img, caption=f"SMILES : {smiles_clean}")
                        inchikey = rdkit.Chem.inchi.MolToInchiKey(mol)
                        smiles_canon = Chem.MolToSmiles(mol)
                        st.markdown(f"**SMILES canonique :** `{smiles_canon}`")
                        st.markdown(f"**InChIKey :** `{inchikey}`")

                    with col_res:
                        st.markdown("#### Propriétés thermodynamiques prédites")
                        c1, c2 = st.columns(2)
                        c1.metric("🌡️ TC — Temp. critique", f"{tc:.2f} K", f"{tc - 273.15:.2f} °C")
                        c2.metric("💨 PC — Pression critique", f"{pc:.2f} bar")
                        c3, c4 = st.columns(2)
                        c3.metric("⚗️ ACEN — Facteur acentrique", f"{acen:.4f}")
                        c4.metric("🌡️ NBP — Point d'ébullition", f"{nbp:.2f} K", f"{nbp - 273.15:.2f} °C")
                        c5, c6 = st.columns(2)
                        c5.metric("❄️ TTP — Point triple", f"{ttr:.2f} K", f"{ttr - 273.15:.2f} °C")
                        c6.metric("💧 VLIQ (Tr=0.8)", f"{vliq:.2f} cm³/mol")

                        st.markdown("---")
                        st.markdown("#### Tableau récapitulatif")
                        df_res = pd.DataFrame({
                            "Propriété": ["TC (K)", "PC (bar)", "ACEN (-)", "NBP (K)", "TTP (K)", "VLIQ (cm³/mol)"],
                            "Valeur prédite": [f"{tc:.4f}", f"{pc:.4f}", f"{acen:.4f}", f"{nbp:.4f}", f"{ttr:.4f}", f"{vliq:.4f}"],
                            "Description": [
                                "Température critique",
                                "Pression critique",
                                "Facteur acentrique",
                                "Point d'ébullition normal",
                                "Point triple",
                                "Volume liq. à Tr=0.8"
                            ]
                        })
                        st.dataframe(df_res, use_container_width=True, hide_index=True)

                        csv = df_res.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="⬇️ Télécharger les résultats (CSV)",
                            data=csv,
                            file_name=f"proprietes_{smiles_clean[:10]}.csv",
                            mime="text/csv"
                        )

                except Exception as e:
                    st.error(f"❌ Erreur lors de la prédiction : {e}")
                    st.info("💡 Vérifiez que tous les fichiers .joblib sont bien dans le même dossier que app.py")

# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 2 : Dessinateur de molécule (Ketcher)
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("✏️ Dessinez votre molécule")
    st.write("Utilisez l'éditeur ci-dessous pour dessiner une molécule. Le SMILES sera généré automatiquement.")

    smiles_ketcher = st_ketcher()

    if smiles_ketcher:
        st.markdown(f"**SMILES généré :** `{smiles_ketcher}`")
        if st.button("🚀 Prédire avec cette molécule", type="primary"):
            valide, mol = valider_smiles(smiles_ketcher)
            if valide:
                with st.spinner("Calcul en cours..."):
                    try:
                        tc, pc, acen, nbp, ttr, vliq = predire_proprietes(smiles_ketcher)
                        st.success("✅ Prédiction réussie !")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("🌡️ TC (K)", f"{tc:.2f}", f"{tc - 273.15:.2f} °C")
                        c2.metric("💨 PC (bar)", f"{pc:.2f}")
                        c3.metric("⚗️ ACEN", f"{acen:.4f}")
                        c4, c5, c6 = st.columns(3)
                        c4.metric("🌡️ NBP (K)", f"{nbp:.2f}", f"{nbp - 273.15:.2f} °C")
                        c5.metric("❄️ TTP (K)", f"{ttr:.2f}", f"{ttr - 273.15:.2f} °C")
                        c6.metric("💧 VLIQ (cm³/mol)", f"{vliq:.2f}")
                    except Exception as e:
                        st.error(f"Erreur : {e}")
            else:
                st.error("SMILES invalide généré par l'éditeur.")

# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 3 : Prédiction en lot par fichier
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("📊 Prédiction en lot — plusieurs molécules")
    st.write("Uploadez un fichier texte avec une colonne SMILES pour prédire les propriétés de plusieurs molécules d'un coup.")

    st.markdown("""
    <div class='warning-box'>
    <b>Format attendu :</b> fichier .txt ou .csv avec une colonne nommée <code>SMILES</code>.
    Séparateur configurable ci-dessous.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

    # Fichier exemple téléchargeable
    Fichier_Exemple = 'SMILES\nC\nN\nC#C\nCCCCCC\n'
    st.download_button('📥 Télécharger un fichier exemple', Fichier_Exemple, 'Liste_Smiles.txt', 'text')

    fichier = st.file_uploader("Choisir un fichier", type=["txt", "csv"])

    col_sep1, col_sep2 = st.columns(2)
    with col_sep1:
        separateur = st.selectbox("Séparateur", [",", ";", "\t", "*"], index=0)
    with col_sep2:
        col_smiles = st.text_input("Nom de la colonne SMILES", value="SMILES")

    if fichier:
        try:
            df_input = pd.read_csv(fichier, sep=separateur)
            st.write(f"✅ Fichier chargé : **{len(df_input)} molécules** détectées")
            st.dataframe(df_input.head(5), use_container_width=True)

            if st.button("🚀 Lancer la prédiction en lot", type="primary"):
                resultats = []
                barre = st.progress(0, text="Prédiction en cours...")

                for i, row in df_input.iterrows():
                    smi = str(row[col_smiles]).strip()
                    valide, _ = valider_smiles(smi)
                    if valide:
                        try:
                            tc, pc, acen, nbp, ttr, vliq = predire_proprietes(smi)
                            resultats.append({
                                "SMILES": smi,
                                "TC (K)": round(tc, 4),
                                "PC (bar)": round(pc, 4),
                                "ACEN (-)": round(acen, 4),
                                "NBP (K)": round(nbp, 4),
                                "TTP (K)": round(ttr, 4),
                                "VLIQ (cm³/mol)": round(vliq, 4),
                                "Statut": "✅ OK"
                            })
                        except:
                            resultats.append({
                                "SMILES": smi, "TC (K)": "-", "PC (bar)": "-",
                                "ACEN (-)": "-", "NBP (K)": "-", "TTP (K)": "-",
                                "VLIQ (cm³/mol)": "-", "Statut": "❌ Erreur"
                            })
                    else:
                        resultats.append({
                            "SMILES": smi, "TC (K)": "-", "PC (bar)": "-",
                            "ACEN (-)": "-", "NBP (K)": "-", "TTP (K)": "-",
                            "VLIQ (cm³/mol)": "-", "Statut": "❌ SMILES invalide"
                        })

                    barre.progress((i + 1) / len(df_input), text=f"Molécule {i+1}/{len(df_input)}")

                barre.empty()
                df_resultats = pd.DataFrame(resultats)
                st.success(f"✅ Prédiction terminée pour {len(df_resultats)} molécules !")
                st.dataframe(df_resultats, use_container_width=True, hide_index=True)

                csv_out = df_resultats.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Télécharger tous les résultats (CSV)",
                    data=csv_out,
                    file_name="resultats_prediction_lrgp.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier : {e}")
            st.info("Vérifiez le séparateur et le nom de la colonne SMILES.")

# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 4 : EOS — Équation d'état (Volume-translated Peng-Robinson)
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("⚗️ EOS — Volume-translated Peng-Robinson + Soave alpha function")
    st.write(
        "Cet onglet calcule et trace les courbes thermodynamiques à partir des propriétés prédites par l'IA. "
        "Entrez un SMILES ou dessinez une molécule pour lancer le calcul EOS."
    )

    st.markdown("""
    <div class='warning-box'>
    ⚠️ <b>Note :</b> L'onglet EOS est disponible uniquement pour les molécules saisies via SMILES ou via le dessinateur de molécules ci-dessous.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

    eos_mode = st.radio(
        "Mode de saisie :",
        ["📝 Entrer un SMILES", "✏️ Dessiner une molécule"],
        horizontal=True
    )

    smiles_eos_clean = ""

    if eos_mode == "📝 Entrer un SMILES":
        # Même pattern que l'onglet 1 : session_state + key uniquement
        if "smiles_eos_val" not in st.session_state:
            st.session_state["smiles_eos_val"] = ""

        col_eos_ex = st.columns([1, 1, 1, 1])
        with col_eos_ex[0]:
            if st.button("💡 Hexane", key="eos_ex_hexane", use_container_width=True):
                st.session_state["smiles_eos_val"] = "CCCCCC"
        with col_eos_ex[1]:
            if st.button("💡 Éthanol", key="eos_ex_ethanol", use_container_width=True):
                st.session_state["smiles_eos_val"] = "CCO"
        with col_eos_ex[2]:
            if st.button("💡 Benzène", key="eos_ex_benzene", use_container_width=True):
                st.session_state["smiles_eos_val"] = "c1ccccc1"
        with col_eos_ex[3]:
            if st.button("💡 Butane", key="eos_ex_butane", use_container_width=True):
                st.session_state["smiles_eos_val"] = "CCCC"

        smiles_eos_input = st.text_input(
            "Notation SMILES pour l'EOS",
            key="smiles_eos_val",
            placeholder="Ex: CCCCCC (hexane), CCO (éthanol)"
        )
        smiles_eos_clean = smiles_eos_input.strip() if smiles_eos_input else ""

    else:
        st.write("Dessinez votre molécule :")
        smiles_eos_drawn = st_ketcher(key="eos_ketcher")
        if smiles_eos_drawn:
            smiles_eos_clean = smiles_eos_drawn.strip()
            st.markdown(f"**SMILES généré :** `{smiles_eos_clean}`")

    lancer_eos = st.button("🚀 Calculer les courbes EOS", type="primary")

    if lancer_eos and smiles_eos_clean:
        valide, mol_eos = valider_smiles(smiles_eos_clean)
        if not valide:
            st.error("❌ SMILES invalide.")
        else:
            atome_interdit, sym = verifier_symboles(smiles_eos_clean)
            if atome_interdit:
                st.warning(f"⚠️ Atome non supporté détecté : `{sym}`")
            else:
                with st.spinner("⏳ Prédiction des propriétés et calcul EOS en cours..."):
                    try:
                        tc, pc, acen, nbp, ttr, vliq = predire_proprietes(smiles_eos_clean)

                        if tc == 88888:
                            st.error("❌ Molécule non supportée par les modèles IA.")
                        else:
                            st.success("✅ Propriétés prédites et courbes EOS calculées !")
                            st.markdown("---")

                            # Résumé des propriétés utilisées
                            st.markdown("#### Propriétés thermodynamiques utilisées pour l'EOS")
                            col_mol_eos, col_prop_eos = st.columns([1, 2])

                            with col_mol_eos:
                                img_eos = Draw.MolToImage(mol_eos, size=(250, 200))
                                st.image(img_eos, caption=f"SMILES : {smiles_eos_clean}")

                            with col_prop_eos:
                                ca, cb, cc = st.columns(3)
                                ca.metric("🌡️ TC (K)", f"{tc:.2f}")
                                cb.metric("💨 PC (bar)", f"{pc:.2f}")
                                cc.metric("⚗️ ACEN", f"{acen:.4f}")
                                cd, ce, _ = st.columns(3)
                                cd.metric("🌡️ NBP (K)", f"{nbp:.2f}")
                                ce.metric("💧 VLIQ Tr=0.8 (cm³/mol)", f"{vliq:.2f}")

                            st.markdown("---")
                            st.markdown("#### Courbes thermodynamiques (EOS)")

                            # Calcul et affichage des 4 graphiques
                            fig1, fig2, fig3, fig4 = tracer_courbes_eos(tc, pc, acen, vliq)

                            col_g1, col_g2 = st.columns(2)
                            with col_g1:
                                st.pyplot(fig1)
                                st.caption("Courbe de pression de vapeur saturante $(P, T)$")
                            with col_g2:
                                st.pyplot(fig2)
                                st.caption("Courbe d'équilibre liquide-vapeur $(T, \\rho)$")

                            col_g3, col_g4 = st.columns(2)
                            with col_g3:
                                st.pyplot(fig3)
                                st.caption("Enthalpie de vaporisation $\\Delta_{vap}H$")
                            with col_g4:
                                st.pyplot(fig4)
                                st.caption("Capacité calorifique du liquide $c_{P,liq}$")

                            plt.close('all')

                    except Exception as e:
                        st.error(f"❌ Erreur lors du calcul EOS : {e}")
                        st.info("Vérifiez que les fichiers .joblib sont disponibles et que le SMILES est valide.")

######################################################################################################
## Pied de page
######################################################################################################

st.markdown("---")
st.markdown(
    "<div class='footer'>© 2026 LRGP — Laboratoire Réactions et Génie des Procédés — UMR CNRS 7274"
    "Université de Lorraine, CNRS, LRGP, F-54000 Nancy, France"
    "Modèles IA développés par Roda Bounaceur — "
    "Fonctions EOS développées par Romain Privat — "
    "Interface IA développée paar Ndeye Diagne</div>",
    unsafe_allow_html=True
)