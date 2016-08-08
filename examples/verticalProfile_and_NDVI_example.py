#!/usr/bin/python

from sellersTwoStream import twoStream
import matplotlib.pyplot as plt
import numpy as np

t=twoStream()
t.setupJULES()

#cosine of solar zenith:
t.mu=1.0
#proportion of diffuse radiation
t.propDif=0.0
#leaf area index
t.lai=5.0
#leaf reflectance & tranmisttance
#these values are for JULES PFT=1 
#and PAR waveband:
t.leaf_r=0.10 #=alpar
t.leaf_t=0.05 #=omega-alpar
#soil reflectance
t.soil_r=0.1
#number of layers
t.nLayers=20

#do the radiative transfer calculation:
IupPAR, IdnPAR, IabPAR, Iab_dLaiPAR = t.getFluxes()

#leaf reflectance & tranmisttance
#these values are for JULES PFT=1 
#and NIR waveband:
t.leaf_r=0.45 #=alnir
t.leaf_t=0.25 #=omnir-alnir

#do the radiative transfer calculation:
IupNIR, IdnNIR, IabNIR, Iab_dLaiNR = t.getFluxes()

#to calculate NDVI use the PAR and NIR albedos
#i.e. the upward flux proportion at the top of 
#the canopy Iup[0] n.b. *lots* of caveats here, 
#not least that this NDVI is calculated from
#a broad band albedo
NDVI=(IupNIR[0]-IupPAR[0])/(IupNIR[0]+IupPAR[0])

#make a nice plot:
plt.plot(IdnPAR[::-1],np.arange(0,t.nLayers+1),'-o',label='PAR')
plt.plot(IdnNIR[::-1],np.arange(0,t.nLayers+1),'-o',label='NIR')
plt.text(0.6, t.nLayers/7., "NDVI=%0.3f"%NDVI, fontsize=22)
plt.xlim([0,1])
plt.ylim([0,t.nLayers])
plt.legend(loc=0)
plt.ylabel('Layer boundary')
plt.xlabel('Downwelling radiation (as a fraction of TOC)')
plt.show()








