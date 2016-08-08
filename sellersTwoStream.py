#!/usr/bin/python
""" sellersTwoStream.py

Implements the Sellers two-stream model of radiative transfer in
vegetation canopies. Capable of being set up to mimic the calculations 
in JULES and CLM as well as having generic functions to allow the use 
of arbitrary leaf angular distributions and "structure factors" 
(after Pinty et al. (2006)).
    
Copyright (C) 2016 Tristan Quaife

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

Tristan Quaife
tquaife@gmail.com
"""
import sys

import numpy as np
import scipy.integrate as integrate

from leafGeometry import leafGeometry

class canopyStructure( ):

  def __init__(self):
    """Various description of the modulation of the
    optical depth as a function of ray geometry and
    canopy structure.
    """
    
    self.pinty_a=0.8
    self.pinty_b=0.1

  def zeta_noStruct(self,mu):
    return 1.0
    
  def zeta_pinty(self,mu):
    return self.pinty_a+self.pinty_b*(1-mu)


class twoStream(leafGeometry,canopyStructure):
  """
  Implements the Sellers' two stream model with 
  various methods to emulate JULES and CLM as well
  as allowing experimenting with other representations.
    
  User defined variables:
    
  self.mu(=1.0)
    Cosine of the solar zenith angle
  self.propDif(=0.0)
    Proportion of diffuse radiation
  self.lai(=5.0)
    Total Leaf Area Index in the canopy 
  self.leaf_r(=0.3)
    Leaf reflectance
  self.leaf_t(=0.2)
    Leaf transmittance
  self.soil_r(=0.1)
    Soil albedo
    
  """

  def __init__(self):
  
    leafGeometry.__init__(self)
    canopyStructure.__init__(self)

    #set up model parameters
    self.mu=1.0
    self.propDif=0.0
    self.leaf_r=0.3
    self.leaf_t=0.2
    self.soil_r=0.1

    self.lai=5.0
    self.nLayers=20
    self.userLayerLAIMap=False
    #intialise the layerMap here just 
    #incase a user sets the flag to 
    #True but forgets to provide a map
    laiPerLayer=self.lai/self.nLayers
    self.layerLAIMap=np.ones(self.nLayers)*laiPerLayer

    
    
    self.setupJULES()


  def setupJULES(self):
    """
    Wire-up the methods so that the flux
    is calculated as in the JULES code
    """

    self.Z=self.zeta_noStruct
    self.G=self.G_JULES
    self.K=self.K_JULES
    self.muBar=self.muBar_JULES
    self.B_direct=self.B_direct_JULES  
    self.B_diffuse=self.B_diffuse_JULES  


  # ========================================
  # K methods ==============================
  # ========================================

  def __K(self):
    """
    Private method containing the common parts
    of the calculation of K
    """
    return self.G(self.mu)/self.mu

  def K_generic(self):
    """
    Optical depth per unit leaf area in the
    direction mu calculated for an arbitrary 
    G/GZ function 
    """
    return self.__K()*self.Z(self.mu)

  def K_JULES(self):
    """
    Optical depth per unit leaf area in the
    direction mu as calculated in JULES
    """
    return self.__K()
  
  def K_CLM(self):
    """
    Optical depth per unit leaf area in the
    direction mu as calculated in CLM
    """
    return self.__K()
  


  # ========================================
  # muBar methods  =========================
  # ========================================


  def muBar_generic(self):
    """
    Average inverse diffuse optical depth per unit leaf area
    calculated for an arbitrary G/GZ function 
    """
    out=integrate.quadrature( self.__muBar_generic_integ,0,1,vec_func=False)
    return out[0] 
    
  def __muBar_generic_integ(self,muDash):
    """
    Private method to be integrated to find muBar
    """
    return muDash/(self.G(muDash)*self.Z(muDash))
  

  def muBar_CLM(self):  
    """
    Average inverse diffuse optical depth per unit leaf area
    as calculated in the CLM implementation 
    """
  
    p1=self.CLM_phi1()
    p2=self.CLM_phi2()
    
    return 1./p2*(1.-p1/p2*np.log((p1+p2)/p1))


  def muBar_JULES(self):
    """
    Average inverse diffuse optical depth per unit leaf area
    as calculated in the JULES implementation 
    
    n.b. is unity in both cases
    """
    
    if self.JULES_lad=='uniform':
      return 1.0
    elif self.JULES_lad=='horizontal':
      return 1.0
    else:
      raise Exception, 'Unknown JULES leaf angle ditribution: '%self.JULES_lad


  # ========================================
  # Volume single scattering albedo ========
  # ========================================


  def volssa_generic(self):
    """
    The volume single scattering albedo for any
    G and Zeta function
    
    See eqn 5 & 7 of Sellers 1985
    """
    out=integrate.quadrature( self.__volssa_generic_integ,0,1,vec_func=False)
    return out[0]*(self.leaf_r+self.leaf_t)*0.5

    
  def __volssa_generic_integ(self,muDash):
    """
    Private method which is integrated to find single scattering albedo
    """
    mu=self.mu
    t1=muDash*(self.G(mu)*self.Z(mu))
    t2=mu*(self.G(muDash)*self.Z(muDash))+muDash*(self.G(mu)*self.Z(mu))

    return t1/t2


  def volssa_CLM(self):
    """
    The volume single scattering albedo as defined
    in CLM. Eqn 3.15 in TechNote 4.
    """
    
    w=self.leaf_r+self.leaf_t
    G=self.G(self.mu)
    
    p1=self.phi1( )
    p2=self.phi2( )
    
    t1=w/2.
    t2=G/(self.mu*p2+G)
    t3=self.mu*p1/(self.mu*p2+G)
    t4=np.log((self.mu*p1+self.mu*p2+G)/(self.mu*p1))

    return t1*t2*(1.-t3*t4)


  def volssa_JULES(self):
    """
    The volume single scattering albedo as defined
    in JULES.     
    """

    w=self.leaf_r+self.leaf_t

    if self.JULES_lad=='uniform':
      return 0.5*w*(1.-self.mu*np.log((self.mu+1.)/self.mu))
    elif self.JULES_lad=='horizontal':
      return w/4.
    else:
      raise Exception, 'Unknown JULES leaf angle ditribution: '%self.JULES_lad




  # ========================================
  # Direct upscatter methods ===============
  # ========================================

  def B_direct_CLM(self):
    """
    Direct upscatter parameter as defined in CLM
    """
    ssa=self.volssa_CLM()
    return self.B_direct_Dickinson(ssa)


  def B_direct_JULES(self):
    """
    Direct upscatter parameter as defined in JULES
    """
    ssa=self.volssa_JULES()
    return self.B_direct_Dickinson(ssa)


  def B_direct_Dickinson_generic_ssa(self):
    """
    Direct upscatter parameter as defined by Dickinson
    but using a generic formulation for the SSA
    """
    ssa=self.volssa_generic()
    return self.B_direct_Dickinson(ssa)


  def B_direct_Dickinson(self,ssa):
    """
    Direct upscatter parameter as defined by Dickinson
    both JULES and CLM use this formulation but differ
    in the calculation of the single scattering albedo.
    """

    w=self.leaf_r+self.leaf_t
    return (1./w)*ssa*(1.+self.muBar()*self.K())/(self.muBar()*self.K())

  
  def B_direct_generic(self):
    """
    Compute the direct upscatter according to Pinty et al. 2006
    (eqn A3)
    """
    w=self.leaf_r+self.leaf_t  
    d=self.leaf_r-self.leaf_t  
    intg=self.integ_cosSq_gDash()

    return (0.5/w)*(w+d*self.mu/self.G(self.mu)*intg)
    
    
  def integ_cosSq_gDash(self):
    """
    Integrate cos^2(theta)*gDash(theta) for calculating 
    upscatter parameters. See Pinty et al. 2006.
    """
    intg=integrate.quad(self.cos2_gDash,0,np.pi/2.)      
    return intg[0]


  def cos2_gDash(self, theta):  
    """ 
    Method to be integrated to find cos^2(theta)*gDash(theta) 
    """
    mu=np.cos(theta)
    return self.gDash(mu)*mu**2



  # ========================================
  # Diffuse upscatter methods ==============
  # ========================================


  def B_diffuse_CLM(self):
    """
    The Diffuse upscatter as calculated in CLM
    as a function of chiL
    """    
    w=self.leaf_r+self.leaf_t
    d=self.leaf_r-self.leaf_t
    
    return (0.5*(w+d*((1.+self.CLM_chiL)/2.)**2.))/w


  def B_diffuse_JULES(self):
    """
    The Diffuse upscatter as calculated in JULES
    """    
    w=self.leaf_r+self.leaf_t
    d=self.leaf_r-self.leaf_t
    if self.JULES_lad=='uniform':
      sqcost=1./3.
    elif self.JULES_lad=='horizontal':
      sqcost=1.0
    else:
      raise Exception, 'Unknown JULES leaf angle ditribution: '%self.JULES_lad
    
    return 0.5*(w+d*sqcost)/w


  def B_diffuse_generic(self):
    """
    Compute the diffuse upscatter according to Pinty et al. 2006
    """
    w=self.leaf_r+self.leaf_t  
    d=self.leaf_r-self.leaf_t  
    intg=self.integ_cosSq_gDash()

    return (0.5/w)*(w+d*intg)


  # ========================================
  # Two stream solution ====================
  # ========================================
  
  
  def getFluxes(self):
    """Calculate the reflected and transmitted
    fluxes using the Sellers 2Stream solution.    
    """
    
    K=self.K()
    muB=self.muBar()
    B_dir=self.B_direct()
    B_dif=self.B_diffuse()
    
    w=self.leaf_r+self.leaf_t
    r_ground=self.soil_r
    
    b=1-w+w*B_dif
    c=w*B_dif
    d=w*B_dir*muB*K
    f=w*muB*K*(1-B_dir)
    h=(np.sqrt(b*b-c*c))/muB
    sigma=(muB*K)**2+c*c-b*b
  
    u1=b-c/r_ground
    u2=b-c*r_ground
    u3=f+c*r_ground
  
    s1=np.exp(-h*self.lai)
    s2=np.exp(-K*self.lai)
  
    p1=b+muB*h
    p2=b-muB*h
    p3=b+muB*K
    p4=b-muB*K
  
    d1=p1*(u1-muB*h)/s1-p2*(u1+muB*h)*s1
    d2=(u2+muB*h)/s1-(u2-muB*h)*s1
  
    #direct up terms:
    h1=-d*p4-c*f
    h2=1./d1*((d-h1/sigma*p3)*(u1-muB*h)/s1-p2*s2*(d-c-h1/sigma*(u1+muB*K)))
    h3=-1./d1*((d-h1/sigma*p3)*(u1+muB*h)*s1-p1*s2*(d-c-h1/sigma*(u1+muB*K)))
    
    #direct down terms
    h4=-f*p3-c*d
    h5=-1./d2*(h4*(u2+muB*h)/(sigma*s1) + (u3-h4/sigma*(u2-muB*K))*s2 )
    h6=1./d2*(h4/sigma*(u2-muB*h)*s1 + (u3-h4/sigma*(u2-muB*K))*s2 )
    
    #diffuse up terms
    h7=c*(u1-muB*h)/(d1*s1)
    h8=-c*s1*(u1+muB*h)/d1
    
    #diffuse down terms
    h9=(u2+muB*h)/(d2*s1)
    h10=-s1*(u2-muB*h)/d2
 
        
    #if not using a user generated lai map
    #generate one here. We do it at this point
    #so as to catch any changes to
    if self.userLayerLAIMap==False:
      laiPerLayer=self.lai/self.nLayers
      self.layerLAIMap=np.ones(self.nLayers)*laiPerLayer
    
    #add a zero layer in (the top of the canopy)
    #also make this a "local scope variable so this
    #doesn't interfere with consecutive calls by adding in
    #multiple zeros
    layerLAIMapPrivate=np.hstack([0.0,self.layerLAIMap])

    #some arrays to hold output
    Iup=np.zeros(len(layerLAIMapPrivate))
    Idn=np.zeros(len(layerLAIMapPrivate))
    Iab=np.zeros(len(layerLAIMapPrivate))
    Iab_dLai=np.zeros(len(layerLAIMapPrivate))

    #calculate fluxes at the top and bottom 
    #of each layer
    laiSum=0 
    for i in xrange(len(layerLAIMapPrivate)):
      
      #keep track of total lai 
      laiSum=laiSum+layerLAIMapPrivate[i]
  
      #fluxes due to collimated irradiance
      Iup_dir=h1*np.exp(-K*laiSum)/sigma + h2*np.exp(-h*laiSum) + h3*np.exp(h*laiSum)
      Idn_dir=h4*np.exp(-K*laiSum)/sigma + h5*np.exp(-h*laiSum) + h6*np.exp(h*laiSum) 
      #add in the collimated radiation to Idn_dir:
      Idn_dir += np.exp(-K*laiSum)

      #fluxes due to diffuse irradiance
      Iup_dif=h7*np.exp(-h*laiSum) +  h8*np.exp(h*laiSum)
      Idn_dif=h9*np.exp(-h*laiSum) + h10*np.exp(h*laiSum)

      #radiation absorbed in layer
      #seloved by energy balance
      #(i.e. fAPAR)      
      if i>0:
        Iab_dir=Iup_dir-Iup_dir_last-Idn_dir+Idn_dir_last
        Iab_dif=Iup_dif-Iup_dif_last-Idn_dif+Idn_dif_last
      else:
        Iab_dir=0.0
        Iab_dif=0.0
      
      #store the values from the layer just
      #calculate for fAPAR calculation
      Iup_dir_last=Iup_dir
      Iup_dif_last=Iup_dif
      Idn_dir_last=Idn_dir
      Idn_dif_last=Idn_dif


      #Calculate absorption per layer using
      #differentiated Sellers equations.
      #This is how fAPAR is calculated in JULES.
      
      if i>0:
        lai=laiSum-layerLAIMapPrivate[i]*.5
        dIup_dir_dLai=-K*h1/sigma*np.exp(-K*lai)-h*h2*np.exp(-h*lai)+h*h3*np.exp(h*lai)
        dIdn_dir_dLai=-K*(h4/sigma+1.)*np.exp(-K*lai)-h*h5*np.exp(-h*lai)+h*h6*np.exp(h*lai)
        dIup_dif_dLai=-h*h7*np.exp(-h*lai)+h*h8*np.exp(h*lai)
        dIdn_dif_dLai=-h*h9*np.exp(-h*lai)+h*h10*np.exp(h*lai)
             
        Iab_dir_dLai=(dIup_dir_dLai-dIdn_dir_dLai)*layerLAIMapPrivate[i]
        Iab_dif_dLai=(dIup_dif_dLai-dIdn_dif_dLai)*layerLAIMapPrivate[i]

      else:
        Iab_dir_dLai=0.0
        Iab_dif_dLai=0.0
                
      #weighted sum of direc/diffuse components
      Iup[i]=(1-self.propDif)*Iup_dir+self.propDif*Iup_dif
      Idn[i]=(1-self.propDif)*Idn_dir+self.propDif*Idn_dif
      Iab[i]=(1-self.propDif)*Iab_dir+self.propDif*Iab_dif
      Iab_dLai[i]=(1-self.propDif)*Iab_dir_dLai+self.propDif*Iab_dif_dLai

    
    return Iup, Idn, Iab, Iab_dLai

    
  
if __name__=="__main__":

  def test():
    print >> sys.stderr, "Write a test function!"

  test()
  
  
