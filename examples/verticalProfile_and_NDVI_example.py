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
t.lai=5.
t.userLayerLAIMap=True
t.layerLAIMap=np.array([0.1,0.3,0.4,0.45,0.5,0.45,0.4,0.3,0.1,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
t.layerLAIMap=t.layerLAIMap*t.lai/t.layerLAIMap.sum()
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
IupNIR, IdnNIR, IabNIR, Iab_dLaiNIR = t.getFluxes()

#now repeat using even LAI 
#(e.g. as would normally be the case in JULES)
t.userLayerLAIMap=False
t.leaf_r=0.10 
t.leaf_t=0.05 
IupPARx, IdnPARx, IabPARx, Iab_dLaiPARx = t.getFluxes()
t.leaf_r=0.45 
t.leaf_t=0.25 
IupNIRx, IdnNIRx, IabNIRx, Iab_dLaiNIRx = t.getFluxes()


#to calculate NDVI use the PAR and NIR albedos
#i.e. the upward flux proportion at the top of 
#the canopy Iup[0] n.b. *lots* of caveats here, 
#not least that this NDVI is calculated from
#a broad band albedo
NDVI=(IupNIR[0]-IupPAR[0])/(IupNIR[0]+IupPAR[0])
NDVIx=(IupNIRx[0]-IupPARx[0])/(IupNIRx[0]+IupPARx[0])

#make a nice plot:
plt.plot(IdnPAR[::-1],np.arange(0,len(t.layerLAIMap)+1),'-o',color="blue",label='PAR - mapped LAI')
plt.plot(IdnNIR[::-1],np.arange(0,len(t.layerLAIMap)+1),'-o',color="green",label='NIR - mapped LAI')
plt.plot(IdnPARx[::-1],np.arange(0,len(t.layerLAIMap)+1),'-x',color="blue",label='PAR - normal LAI')
plt.plot(IdnNIRx[::-1],np.arange(0,len(t.layerLAIMap)+1),'-x',color="green",label='NIR - normal LAI')
plt.text(0.5, t.nLayers/2.5, "NDVI (mapped)=%0.3f"%NDVI, fontsize=22)
plt.text(0.5, t.nLayers/3., "NDVI (normal)=%0.3f"%NDVIx, fontsize=22)
plt.xlim([0,1])
plt.ylim([0,t.nLayers])
plt.legend(loc=0)
plt.ylabel('Layer boundary')
plt.xlabel('Downwelling radiation (as a fraction of TOC)')
plt.show()








