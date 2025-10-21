#
# Copyright (C) 2012-2020 Euclid Science Ground Segment
#
# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3.0 of the License, or (at your option)
# any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
#

"""
File: python/SIR_SelfCalib/SelfCalib.py

Created on: 04/14/20
Author: shemmati, Georgio
"""

import time
from datetime import datetime
import os
import numpy as np
import astropy
from astropy import stats
import astropy.coordinates
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import wcs
from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import sparse
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator as RGI
import math
from scipy import interpolate
import scipy.ndimage as ndimage
pi = math.pi

from astropy import stats #JX
import scipy.stats #JX

#from SIR_SpectraDecontamination.LocationTableSet import LocationTableSet
from SIR_SpectraExtraction.H5ExtractedSpectraCollection import H5ExtractedSpectraCollection 
from SIR_H5SpectraExtractionBinding import H5ExtractedSpectrum
#from SIR_SpectraExtractionBinding import SpectrumDataSet
from SIR_InstrumentModels.DetectorModel import DetectorModel

from SIR_FluxCalibration.Fit2D import *
from SIR_FluxCalibration.FPA_Geometry import *
from SIR_FluxCalibration.Utils import *

#import ST_DataModelBindings.bas.ppr_stub as ppr_dict #JX

from SIR_InstrumentModels.OpticalModel import OpticalModel
from SIR_FluxCalibration.RelativeFluxCorrectionCore import RelativeFluxCorrectionCore

###############################################################################################################
######################### cal mode 2: simulated dithers magnitude match to GAIA ############################### 
###############################################################################################################

class SelfCalibSirGaiaData:
    """A class to store Self-calibration input and temporary data"""
    
    def sir_starcatalog(self,h5file,tables):
        ''' measuring flux in a wavebin for SIR spectra given an 
        h5 file and a location table'''
    
        det_id = h5file.getDetectorID()
        dither = h5file.getDither()
        loc_t = tables
      
        ids,xf,yf =[],[],[]
        xd,yd = [],[]
        xim,yim = [],[]
        jm,hm = [],[]
        chunkm,dc = [],[]
        ifile = []
        detectorss = []
        
        detmod = None
        for spectrum in h5file:
                
            metadata = loc_t[spectrum.getObjectID()].getAstronomicalObject()
            ras,decl = metadata.getCoords()
            if metadata.hasMagnitude('J') is False:
                continue
            if metadata.hasMagnitude('H') is False:
                continue
                        
            j_mag = metadata.getMagnitude('J').getValue()
            h_mag = metadata.getMagnitude('H').getValue()   
            locc = spectrum.getLocationSpectrum()
            xy = locc.computePosition(self.wcenter, row=2).getPosition() ## detector coordinates in pixel

            if detmod is None:
                detmod = locc.getDetectorModel()
                box = detmod.getEnvelopeBox()
                _xpixels = _det_model.getDetXPixels()
                _ypixels = _det_model.getDetYPixels()
                                
            (xdp,ydp) = normaldet(xy,maxx=_xpixels,maxy=_ypixels) ## detector coordinates [-1,1]            
            (xfp,yfp) = normalfov(box, detmod.getFOVPositions(xy, det_id).T) ## focal plane in [-1,1]

        
            flux = spectrum.getSpectrumDataSet(H5ExtractedSpectrum.SPEC1D_LABEL).getActualData()
            var = spectrum.getSpectrumDataSet(H5ExtractedSpectrum.SPEC1D_LABEL).getActualVariance()
            quality = spectrum.getSpectrumDataSet(H5ExtractedSpectrum.SPEC1D_LABEL).getQuality()
            wave = spectrum.getWavelength()    
            #metadata = spectrum.getAstronomicalObject()
            #h_mag = metadata.getMagnitude('H').getValue() ### added this on March 14, 2024
            mm,dmm = measureChunkMagnitude1D(wave,flux,var,quality,center_w = self.wcenter,delta_w=self.deltaw)  
            if (h_mag<self.limmag) and (h_mag>16) and (mm<32) and (xy[0]>0) and (xy[0]<_xpixels) and (xy[1]>0) and (xy[1]<_ypixels):  
                jm.append(j_mag)
                hm.append(h_mag)
                chunkm.append(mm)
                dc.append(dmm)
                ids.append(spectrum.getObjectID())
                xf.append(xfp)
                yf.append(yfp)
                xd.append(xdp)
                yd.append(ydp)
                xim.append(xy[0])
                yim.append(xy[1])
                ifile.append(dither)
                detectorss.append(det_id)
        
        sirr = Table()
        sirr['ID'], sirr['Xfp'], sirr['Yfp'] = np.array(ids),np.array(xf),np.array(yf)
        sirr['Xd'],sirr['Yd'] = np.array(xd),np.array(yd)
        sirr['Ximage'],sirr['Yimage'] = np.array(xim),np.array(yim)
        sirr['jmag'],sirr['hmag'], sirr['chunkm'],sirr['chunkdm'],sirr['ifile'],sirr['detector'] = np.array(jm),np.array(hm),np.array(chunkm),np.array(dc),np.array(ifile),np.array(detectorss) 

        return sirr
                                                                                                                           
    def gaia2sir(self,sir):
        ''' Matching the stars in the NEP from a GAIA catalog down to a limmag 
        to closest (very approximately) sir star'''
        gaia = self.gaia
        sirID,sirxfp,siryfp,sirximage,siryimage,sirxdet,sirydet,sirJ,sirH,sirmag,sirdmag,sirdither,sirdetectors = sir['ID'],sir['Xfp'],sir['Yfp'],sir['Ximage'],sir['Yimage'],sir['Xd'],sir['Yd'],sir['jmag'],sir['hmag'],sir['chunkm'],sir['chunkdm'],sir['ifile'], sir['detector']
        #sir15=(sirJ+sirH)/2.0#+ (sirmag-sirH) ### the SIR star magnitude at 1.5 with a correction spectra vs TU
    
        best_star,starY,starJ,starH,starDetx,starDety = np.zeros_like(gaia['w1']),np.zeros_like(gaia['w1']),np.zeros_like(gaia['w1']),np.zeros_like(gaia['w1']),np.zeros_like(gaia['w1']),np.zeros_like(gaia['w1'])
        starmag,stardmag = np.zeros_like(gaia['w1']),np.zeros_like(gaia['w1'])
        stardither = np.zeros_like(gaia['w1'])
        starFPx,starFPy,starimagex,starimagey = np.zeros_like(gaia['w1']),np.zeros_like(gaia['w1']),np.zeros_like(gaia['w1']),np.zeros_like(gaia['w1'])
        stardetector = np.zeros_like(gaia['w1'])
        for s in range(len(gaia['ra'])):
            if (flux2mag(gaia['rp'][s])>14)and(flux2mag(gaia['rp'][s])<self.limmag):
                #y_g_int = np.interp(1.5,[0.5,3.4],[gaia['rp'][s],gaia['w1'][s]])
                #y_g = flux2mag(5.0*y_g_int)
                y_g = 1.032*flux2mag(gaia['w1'][s])-1.55
                u = np.where(sirJ == min(sirJ, key=lambda x:abs(x-y_g)))[0][0]
                best_star[s] = sirID[u]
                starJ[s] = sirJ[u]
                starH[s] = sirH[u]
                starFPx[s] = sirxfp[u]
                starFPy[s] = siryfp[u]
                starDetx[s] = sirxdet[u]
                starDety[s] = sirydet [u]
                starimagex[s] = sirximage[u]
                starimagey[s] = siryimage[u]
                starmag[s],stardmag[s] = sirmag[u],sirdmag[u]
                stardither[s] = sirdither[u]
                stardetector[s] = sirdetectors[u]

        u = (starFPx!=0)
        nn = len(starFPx[u])
        best_star1,starY1,starJ1,starH1,starFPx1,starFPy1,starmag1,stardmag1,stardither1 = np.zeros(nn),np.zeros(nn),np.zeros(nn),np.zeros(nn),np.zeros(nn),np.zeros(nn),np.zeros(nn),np.zeros(nn),np.zeros(nn)
        
        starimagex1,starimagey1,starDetx1,starDety1,stardetector1 = np.zeros(nn),np.zeros(nn),np.zeros(nn),np.zeros(nn),np.zeros(nn)
        
        best_star1,starY1,starJ1,starH1,starFPx1,starFPy1,starmag1,stardmag1,stardither1 = best_star[u],starY[u],starJ[u],starH[u],starFPx[u],starFPy[u],starmag[u],stardmag[u],stardither[u]
        
        starimagex1,starimagey1,starDetx1,starDety1,stardetector1 = starimagex[u],starimagey[u],starDetx[u],starDety[u],stardetector[u]
        
        gra,gdec,grp,gw1 = gaia['ra'][u],gaia['dec'][u],gaia['rp'][u],gaia['w1'][u]

        gaiasir = Table()
        gaiasir['SIR_ID'], gaiasir['starXfp'], gaiasir['starYfp'] = np.arange(len(starDetx1)),starFPx1,starFPy1
        gaiasir['starXdet'],gaiasir['starYdet'],gaiasir['starXimage'],gaiasir['starYimage'] =starDetx1,starDety1,starimagex1,starimagey1
        gaiasir['mag'],gaiasir['dmag'] = starmag1,stardmag1
        gaiasir['ra'], gaiasir['dec'] = gra, gdec
        gaiasir['ifile']=stardither1
        gaiasir['detector']=stardetector1
    
        return gaiasir

    def dithergaia(self,go,numdither=15,outdir='.'):
        ''' Some survey strategy over the NEP to have stars 
        with nobs~ 1-16 and then change their magnitude based
        on an input flat cube'''
        
        c = SkyCoord(go['ra'], go['dec'], unit='deg')
        ras = []
        decs = []
        #rots = []
        rollangle = np.random.uniform(-180,180)
        
        while len(ras) < numdither:
            randra = np.random.uniform(266.5,270.5)
            randdec = np.random.uniform(64.75,66.25)
    
            if np.sqrt(((randra-268.625)*np.cos(randdec*pi/180))**2 + (randdec-65.6)**2) < 0.6:
                ras.append(randra)
                decs.append(randdec)
                #rots.append(np.random.uniform(-180,180))
        # survey part

        obs = np.zeros_like(go['ra'])
        nid,nra,ndec,nmag,ndmag,nxfp,nyfp,nifile = [],[],[],[],[],[],[],[]
        nxdet,nydet,nxim,nyim = [],[],[],[]
        newX,newY = [],[] 
        ndetector = []
        fig = plt.figure()
        plt.plot(go['ra'],go['dec'],'.',color='gray',alpha=0.1)

        for i in range(len(ras)):
            testw = get_euclid_wcs(ras[i],decs[i],roll=rollangle)

            #Get pixel coordinates of each star, convert to FOV coordinate (-1 to 1 in both dimensions)
            px, py = testw.world_to_pixel(c)
            fovx = px/1e4
            fovy = py/1e4

            ss = np.where((np.abs(fovx)<1) & (np.abs(fovy)<1))
            stars_in_pointing = go[ss]

            ra2,dec2 = go['ra'][ss],go['dec'][ss]
            sirid,sdetx,sdety = go['SIR_ID'][ss],go['starXdet'][ss],go['starYdet'][ss]
            sfpx,sfpy = go['starXfp'][ss],go['starYfp'][ss]
            simx,simy = go['starXimage'][ss],go['starYimage'][ss]
            nx,ny = fovx[ss],fovy[ss]
            smag,sdmag = go['mag'][ss],go['dmag'][ss]
            sdither = go['ifile'][ss]
            sdetector = go['detector'][ss]
            plt.plot(go['ra'][ss],go['dec'][ss],'.',color=plt.cm.viridis(i*int(256/len(ras))))        

            for s in range(len(ra2)):
                nra.extend([ra2[s]])
                ndec.extend([dec2[s]])
                nid.extend([sirid[s]])
                nxfp.extend([sfpx[s]])
                nyfp.extend([sfpy[s]])
                nxdet.extend([sdetx[s]])
                nydet.extend([sdety[s]])
                nxim.extend([simx[s]])
                nyim.extend([simx[s]])
                nmag.extend([smag[s]])
                ndmag.extend([sdmag[s]])             
                newX.extend([nx[s]])
                newY.extend([ny[s]])
                nifile.extend([sdither[s]])
                ndetector.extend([sdetector[s]])
                
        plt.xlabel("RA")
        plt.ylabel("DEC")
        plt.savefig(os.path.join(outdir, "02-RADEC.png"), dpi=200)
    
        # Change of magnitudes
        X =  np.linspace(-1.,1.,np.shape(self.cube)[1])
        Y =  np.linspace(-1.,1.,np.shape(self.cube)[2])
        Z =  np.linspace(self.wstart,self.wend,np.shape(self.cube)[0]) #JX

        fn = RGI((Z,X,Y), self.cube) ## interpolating on the response
        mag_new,dmag_new = np.zeros_like(nmag),np.zeros_like(ndmag)
        ndet = np.zeros_like(nmag)
        
        for k in range(len(nra)):
            flux = 10.0**((nmag[k]-25.0)/-2.5)#mag[i]#
            pts_old = np.array([[self.wcenter,nxfp[k],nyfp[k]]])
            pts_new = np.array([[self.wcenter,newX[k],newY[k]]]) 
            mag_new[k] = -2.5*np.log10(flux * (10.0**(fn(pts_new)/-2.5))/(10.0**(fn(pts_old)/-2.5)))+25.0+ self.noisefactor*np.random.normal(0.,ndmag[k])


        # saving with the format readable by the rest of SIR self calib data
        gaiasirdither = Table()
        gaiasirdither['OBJECT_ID'], gaiasirdither['X_fp'], gaiasirdither['Y_fp'] = nid,newX,newY
        gaiasirdither['MAG_APER'],gaiasirdither['MAGERR_APER'] = mag_new,ndmag+0.0001*np.random.lognormal(0,1,len(ndmag))
        gaiasirdither['ALPHA_J2000'], gaiasirdither['DELTA_J2000'] = nra, ndec
        gaiasirdither['FLAGS_EXTRACTION'], gaiasirdither['FLAGS_SCAMP'], gaiasirdither['FLAGS_IMA'] = np.zeros_like(nra),np.zeros_like(nra),np.zeros_like(nra)
        gaiasirdither['ifile'], gaiasirdither['idet'] = np.array(nifile).astype(int),ndetector
        gaiasirdither['X_IMAGE'], gaiasirdither['Y_IMAGE'] = nxim, nyim
        gaiasirdither['X_det'], gaiasirdither['Y_det'] = nxdet, nydet
    
        return gaiasirdither

    def extractStarCatalog(self, obs):
        # Prepare catalog of individual stars
        catalog = Table()
        catalog['OBJECT_ID'] = np.unique(obs['OBJECT_ID'])
        catalog['nobs'] = np.repeat(     0, len(catalog))
        catalog['mag']  = np.repeat(np.NaN, len(catalog))
        catalog['unc']  = np.repeat(np.NaN, len(catalog))

        index_star2obs = []
        obs['istar'] = -1

        for i in range(0, len(catalog)):
            j = np.where(obs['OBJECT_ID'] == catalog['OBJECT_ID'][i])[0]
            index_star2obs.append(j)
        index_star2obs = np.asarray(index_star2obs)

        for i in range(0, len(catalog)):
            j = index_star2obs[i]
            catalog['nobs'][i] = j.size
            catalog['mag'][ i] = np.mean(obs['MAG_APER'][j])
            catalog['unc'][ i] = np.std( obs['MAG_APER'][j])
            obs['istar'][j] = i

        index_det2obs = []
        for idet in range(0, 16):
            i = np.where(obs['idet'] == (idet+1))[0]
            index_det2obs.append(i)
        index_det2obs = np.asarray(index_det2obs)
        #assert np.min(catalog['nobs']) == 2
        return (catalog, index_star2obs, index_det2obs)


    def __init__(self, extspecnames, tables, ncoeff, limmag, wcenter, deltaw, grism_param, noisefactor, ndither, mdb, outdir="."):
        
        obs = ()

        dataset_name = H5ExtractedSpectrum.SPEC1D_LABEL

        if outdir is not None:
            if not os.path.isdir(outdir):
                os.mkdir(outdir)

        self.ncoeff = ncoeff
        self.limmag = limmag
        self.outdir = outdir
        self.wcenter = wcenter
        self.deltaw = deltaw
        self.noisefactor = noisefactor
        self.ndither = ndither
        self.mdb = mdb

        #JX
        exspectra_test = H5ExtractedSpectraCollection.load(extspecnames[0], ids=[H5ExtractedSpectrum.SPEC1D_LABEL,])
        spectra = exspectra_test.getExtractedSpectra()
        wavetest = spectra[0].getWavelength()
        self.wstart = wavetest[0]
        self.wend = wavetest[len(wavetest)-1]

        fakecube_name = os.path.join(self.outdir, 'data', 'EUC_SIR_W-RELATIVEFLUX-SCALE_3_manual_newDM_2022Nov29.fits') 
        self.cube = fits.getdata(fakecube_name,1) # all 0 grism# vignetted tilted grism
                
        gaia_name = os.path.join(self.outdir, 'data', 'euclid_selfcal_stars_gaia_allwise_2021july26.fits') 
        self.gaia = fits.getdata(gaia_name,1)

        # Read input files
        print("Loading spectra")
        sirr = ()
        for k,filename in enumerate(extspecnames): 
            if (tables[k].getGWAPosition()+tables[k].getGWATilt()) == grism_param:
                exspectra = H5ExtractedSpectraCollection.load(filename, ids=[H5ExtractedSpectrum.SPEC1D_LABEL,])
                print ("Loading file ", filename) 

                sirr_one = self.sir_starcatalog (exspectra,tables[k])
                if len(sirr) ==0:
                    sirr = sirr_one.copy()
                else:
                    sirr = astropy.table.vstack([sirr,sirr_one])
        
        gaiasir = self.gaia2sir(sirr)
        data = self.dithergaia(gaiasir,numdither=self.ndither, outdir = self.outdir)
       
        if len(obs) == 0:
            obs = data.copy()
        else:
            obs = astropy.table.vstack([obs, data])

        nn = len(obs)
        # Sort by star magnitude
        i = np.argsort(obs['MAG_APER'])[::-1]
        obs = obs[i]
        (catalog, index_star2obs, index_det2obs) = self.extractStarCatalog(obs)

        self.obs = obs
        self.catalog = catalog
        self.index_star2obs = index_star2obs
        self.index_det2obs = index_det2obs

        print("Number of coefficients in the model                : ", self.ncoeff)
        print("Number of individual stars                         : ", len(catalog))
        print("Number of constraints                              : ", len(obs))
 
        if outdir is not None:
            fig = plt.figure()
            plt.plot(obs['MAG_APER'], obs['MAGERR_APER'], marker='.', linestyle="None", ms=1.5)
            plt.xlabel("Magnitude")
            plt.ylabel("Mag. uncertainty")
            plt.yscale("log")
            plt.savefig(os.path.join(outdir, "03-SourceMagAndUnc.png"), dpi=200)
            
            fig = plt.figure()
            plt.plot(obs['ALPHA_J2000'], obs['DELTA_J2000'], marker='.', linestyle="None", ms=1.5)
            plt.xlabel("RA")
            plt.ylabel("DEC")
            plt.savefig(os.path.join(outdir, "01-RADEC.png"), dpi=200)

###############################################################################################################
###################################### cal mode 1: random dithered observations ############################### 
###############################################################################################################

class SelfDitherData:
    """To manually dither and generate more data"""
    
    
    def extractStarCatalog2(self, obs):
        # Prepare catalog of individual stars
        catalog = Table()
        catalog['OBJECT_ID'] = np.unique(obs['OBJECT_ID'])
        catalog['nobs'] = np.repeat(     0, len(catalog))
        catalog['mag']  = np.repeat(np.NaN, len(catalog))
        catalog['unc']  = np.repeat(np.NaN, len(catalog))

        index_star2obs = []
        obs['istar'] = -1

        for i in range(0, len(catalog)):
            j = np.where(obs['OBJECT_ID'] == catalog['OBJECT_ID'][i])[0]
            index_star2obs.append(j)
        index_star2obs = np.asarray(index_star2obs)

        for i in range(0, len(catalog)):
            j = index_star2obs[i]
            catalog['nobs'][i] = j.size
            catalog['mag'][ i] = np.mean(obs['MAG_APER'][j])
            catalog['unc'][ i] = np.std( obs['MAG_APER'][j])
            obs['istar'][j] = i

        index_det2obs = []
        for idet in range(0, 16):
            i = np.where(obs['idet'] == (idet+1))[0]
            index_det2obs.append(i)
        index_det2obs = np.asarray(index_det2obs)
        #assert np.min(catalog['nobs']) == 2
        return (catalog, index_star2obs, index_det2obs)
    
    def __init__(self, data, mdb, gwapos, ndithers = 10, noisefactor = 1):
        self.ncoeff = data.ncoeff
        self.limmag = data.limmag
        self.outdir = data.outdir
        self.data = data
        self.wcenter = data.wcenter
        self.ndithers = ndithers
        self.noisefactor = noisefactor
        fakecube_name = os.path.join(self.outdir, 'data', 'EUC_SIR_W-RELATIVEFLUX-SCALE_3_manual_newDM_2022Nov29.fits') 
        self.cube = fits.getdata(fakecube_name,1) # a vignetted tilted grism
        #self.cube = illuminate() 
        detmod = DetectorModel(gwapos, mdb, rotate=True) # change false to True once new data
        box = detmod.getEnvelopeBox()

        idd = data.obs['OBJECT_ID']
        #ra,dec = data.obs['ALPHA_J2000'],data.obs['DELTA_J2000']
        xfp,yfp = data.obs['X_fp'], data.obs['Y_fp']
        mag,magerr = data.obs['MAG_APER'], data.obs['MAGERR_APER']

        sidd,sximage,syimage,sxd,syd,sxf,syf,smag,smagerr,sflag1,sflag2,sflag3,sdet,sdither = [],[],[],[],[],[],[],[],[],[],[],[],[],[]
        obs2 = ()
        
        X =  np.linspace(-1.,1.,np.shape(self.cube)[1])
        Y =  np.linspace(-1.,1.,np.shape(self.cube)[2])
        if self.wcenter<12500:
            #Z =  np.linspace(8548.5,13862.1,np.shape(self.cube)[0])
            Z =  np.linspace(9000.0,13862.1,np.shape(self.cube)[0])
            
        else:
            Z =  np.linspace(11900.0,19002.0,np.shape(self.cube)[0]) 
            
        fn = RGI((Z,X,Y), self.cube) ## interpolating on the response

        print("Dithering spectra")

        for i in range(len(xfp)):
            #det_new = np.random.randint(0,16,self.ndithers) # which detector
            #det_id = ['11','12','13','14','21','22','23','24','31','32','33','34','41','42','43','44']

            flux = 10.0**((mag[i]-25.0)/-2.5)#mag[i]#
            j=0
            while j < self.ndithers:
                x_new = np.random.uniform(-1,1,1) #normalized fp x 
                y_new = np.random.uniform(-1,1,1) #normalized fp y
                fov = normalfov_r(box,np.array([x_new,y_new])) #fov coordinates in mm
                pix = detmod.getPixel(fov[0],fov[1])  # pixel coordinates
                if pix[0]!=0:
                    detid = pix.getDetectorNumber()
                    pts_old = np.array([[self.wcenter,xfp[i],yfp[i]]])
                    pts_new = np.array([[self.wcenter,xf,yf]]) 
                    mag_new = -2.5*np.log10(flux * (10.0**(fn(pts_new)/-2.5))/(10.0**(fn(pts_old)/-2.5)))+25.0
                    sidd.append(idd[i])
                    sximage.append(fov[0]) #in mm not sure
                    syimage.append(fov[1]) #in mm
                    #sdet.append(det_id[det_new[j]])
                    sdet.append(detid)
                    sdither.append(j)
                    sxd.append(xd)
                    syd.append(yd)
                    sxf.append(x_new)
                    syf.append(y_new)
                    #sra.append(ra[i])
                    #sdec.append(dec[i])
                    smag.append(mag_new[0]+np.random.normal(0.,magerr[i]*self.noisefactor)) # added some noise 
                    smagerr.append(magerr[i])
                    sflag1.append(0)
                    sflag2.append(0)
                    sflag3.append(0)
                    j+=1

        data = Table()
        data['OBJECT_ID'],data['X_IMAGE'],data['Y_IMAGE'], data['ALPHA_J2000'],data['DELTA_J2000'] = sidd,sximage,syimage,sra,sdec
        data['MAG_APER'],data['MAGERR_APER'] = smag, smagerr
        data['FLAGS_EXTRACTION'], data['FLAGS_SCAMP'],data['FLAGS_IMA'] = sflag1,sflag2,sflag3
            
        c = astropy.table.Column(name='ifile', data=sdither); data.add_column(c)
        c = astropy.table.Column(name='idet' , data=sdet); data.add_column(c)
        #(x, y) = pixel2focalplane(data['idet'], data['X_IMAGE'], data['Y_IMAGE'])
        c = astropy.table.Column(name='X_fp' , data=sxf); data.add_column(c)
        c = astropy.table.Column(name='Y_fp' , data=syf); data.add_column(c)

        # Trasform x and y coordinates into the [-1,1] range on the detector
        #(x, y) = pixel2det(data['idet'], data['X_IMAGE'], data['Y_IMAGE'])
        c = astropy.table.Column(name='X_det' , data=sxd); data.add_column(c)
        c = astropy.table.Column(name='Y_det' , data=syd); data.add_column(c)
    
        if len(obs2) == 0:
            obs2 = data.copy()
        else:
            obs2 = astropy.table.vstack([obs2, data])
            
        i = np.argsort(obs2['MAG_APER'])[::-1]
        obs2 = obs2[i]
        (catalog, index_star2obs, index_det2obs) = self.extractStarCatalog2(obs2)

        self.obs = obs2
        self.catalog = catalog
        self.index_star2obs = index_star2obs
        self.index_det2obs = index_det2obs
                    
###############################################################################################################
###################################### cal mode 0: main mode for selfcal observations ######################### 
###############################################################################################################

class SelfCalibSirData:
    """A class to store Self-calibration input and temporary data"""
    
    def remove_unique_object_ids(self,data):
        unique, counts = np.unique(data['OBJECT_ID'], return_counts=True)
        repeated_ids = unique[counts > 1]
        return data[np.isin(data['OBJECT_ID'], repeated_ids)]

    def sigmaclip(self,data):
        masks = []
        object_ids = np.unique(data['OBJECT_ID'])
        for object_id in object_ids:
            current_entries = data[data['OBJECT_ID'] == object_id]
            median_mag_aper = np.median(current_entries['MAG_APER'])
            std_mag_aper = np.std(current_entries['MAG_APER'])
            mask = (current_entries['MAG_APER'] >= median_mag_aper - 3.0*std_mag_aper) & (current_entries['MAG_APER'] <= median_mag_aper + 3.0*std_mag_aper)
            masks.append(mask)
        final_mask = np.hstack(masks)
        return data[final_mask]
    
    def extractStarCatalog(self, obs):
        # Prepare catalog of individual stars
        catalog = Table()
        catalog['OBJECT_ID'] = np.unique(obs['OBJECT_ID'])
        catalog['nobs'] = np.repeat(     0, len(catalog))
        catalog['mag']  = np.repeat(np.NaN, len(catalog))
        catalog['unc']  = np.repeat(np.NaN, len(catalog))

        index_star2obs = []
        obs['istar'] = -1

        for i in range(0, len(catalog)):
            j = np.where(obs['OBJECT_ID'] == catalog['OBJECT_ID'][i])[0]
            index_star2obs.append(j)
        #JX index_star2obs = np.asarray(index_star2obs) #JX 10-25-2024: Added dtype=object to remove warning message.
        index_star2obs = np.asarray(index_star2obs, dtype=object)

        for i in range(0, len(catalog)):
            j = index_star2obs[i]
            catalog['nobs'][i] = j.size
            catalog['mag'][ i] = np.mean(obs['MAG_APER'][j])
            catalog['unc'][ i] = np.mean(obs['MAGERR_APER'][j])#np.std( obs['MAG_APER'][j])
            obs['istar'][j] = i

        index_det2obs = []
        for idet in range(0, 16):
            i = np.where(obs['idet'] == (idet+1))[0]
            index_det2obs.append(i)
        #JX index_det2obs = np.asarray(index_det2obs) #JX 10-25-2024: Added dtype=object to remove warning message.
        index_det2obs = np.asarray(index_det2obs, dtype=object)

        #assert np.min(catalog['nobs']) == 2
        return (catalog, index_star2obs, index_det2obs)


    def __init__(self, extspecnames,listallgrisms, ncoeff, limmag, wcenter, deltaw, grism_param, mdb, outdir="."):
        
        #JX 10-25-2024 SpeedUp: Created the following lists.
        #obs = ()
        obs = [[] for _ in range(len(wcenter))]
        catalog = [[] for _ in range(len(wcenter))]
        index_star2obs = [[] for _ in range(len(wcenter))]
        index_det2obs = [[] for _ in range(len(wcenter))]
        #JXJXJX

        dataset_name = H5ExtractedSpectrum.SPEC1D_LABEL

        self.ncoeff = ncoeff
        self.limmag = limmag
        self.outdir = outdir
        self.wcenter = wcenter
        self.deltaw = deltaw
        self.mdb = mdb

        if self.outdir is not None:
            self.outdir = os.path.join(self.outdir, 'aux')
            os.makedirs(self.outdir, exist_ok=True)

        # Read input files
        print("Loading spectra")
        for k,filename in enumerate(extspecnames): 
            if listallgrisms[k]==grism_param:
                try:
                    #exspectra = H5ExtractedSpectraCollection.load(filename, ids=[H5ExtractedSpectrum.SPEC1D_LABEL,RelativeFluxCorrectionCore.SPEC1D_RELFLUX_LABEL, ])
                    exspectra = H5ExtractedSpectraCollection.load(filename, ids=[H5ExtractedSpectrum.SPEC1D_LABEL, ])
                except:
                    print('no 1d rel?')
                    continue
            #if (exspectra.getGWAPosition()+exspectra.getGWATilt()) == grism_param:
                idet = exspectra.getDetectorNumber()
                det_id = exspectra.getDetectorID()
                
                dither = exspectra.getDither()
                #loc_t = tables[k]
      
                idd,ximage,yimage,ra,dec,mag,magerr,flag1,flag2,flag3 = [],[],[],[],[],[],[],[],[],[]
                xfp,yfp,xdn,ydn = [],[],[],[]
                
                detmod = None 
                #detmod = DetectorModel(exspectra.getGWAPositionName(), mdb, rotate=True)

                #JX 10-25-2024 SpeedUp: Created the loop on n
                for n in range(len(wcenter)):

                    for spectrum in exspectra:   
                        #metadata = loc_t[spectrum.getObjectID()].getAstronomicalObject()
                        #ras,decl = metadata.getCoords()
                        #h_mag = metadata.getMagnitude('H').getValue()
                        #### new
                        locc = spectrum.getLocationSpectrum()
                        xy = locc.computePosition(wcenter[n], row=2).getPosition() ## detector coordinates in pixel

                        if detmod is None:
                            detmod = locc.getDetectorModel()
                            box = detmod.getEnvelopeBox()
                            _xpixels = detmod.getDetXPixels()
                            _ypixels = detmod.getDetYPixels()

                        try:
                            #flux = spectrum.getSpectrumDataSet(RelativeFluxCorrectionCore.SPEC1D_RELFLUX_LABEL).getActualData()
                            #var = spectrum.getSpectrumDataSet(RelativeFluxCorrectionCore.SPEC1D_RELFLUX_LABEL).getActualVariance()
                            #quality = spectrum.getSpectrumDataSet(H5ExtractedSpectrum.SPEC1D_LABEL).getQuality()
                            flux = spectrum.getSpectrumDataSet(dataset_name).getActualData()
                            var = spectrum.getSpectrumDataSet(dataset_name).getActualVariance()
                            quality = spectrum.getSpectrumDataSet(dataset_name).getQuality()
                            wave = spectrum.getWavelength()    
                        except:
                            print("Coudn't use spectrum")
                            continue

                        metadata = spectrum.getAstronomicalObject()
                        if metadata.hasMagnitude('H') is False:
                            continue
                        h_mag = metadata.getMagnitude('H').getValue()
                        #### and (spectrum.getObjectID()%2==0)
                                            
                        if (h_mag>16) and(h_mag<self.limmag) and (xy[0]>0) and (xy[0]<_xpixels) and (xy[1]>0) and (xy[1]<_ypixels):
                            m,dm = measureChunkMagnitude1D(wave,flux,var,quality,self.wcenter[n],self.deltaw)    
                            (xd,yd) = normaldet(xy,maxx=_xpixels, maxy=_ypixels) ## detector coordinates [-1,1]
                            (xf,yf) = normalfov(box,detmod.getFOVPositions(xy, det_id).T) ## focal plane in [-1,1]

                            #snr = 1/(1 - np.power(10,-0.4*dm)) dm<0.1 == snr>10
                            if dm<0.1:
                                #if m>-90:
                                idd.append(spectrum.getObjectID())
                                ximage.append(xy[0])
                                yimage.append(xy[1])
                                xdn.append(xd)
                                ydn.append(yd)
                                xfp.append(xf)
                                yfp.append(yf)
                                #ra.append(ras)
                                #dec.append(decl)
                                mag.append(m)
                                magerr.append(dm)
                                flag1.append(0)
                                flag2.append(0)
                                flag3.append(0)

                    data = Table()
                    #data['OBJECT_ID'],data['X_IMAGE'],data['Y_IMAGE'], data['ALPHA_J2000'],data['DELTA_J2000'] = idd,ximage,yimage,ra,dec
                    data['OBJECT_ID'],data['X_IMAGE'],data['Y_IMAGE'] = idd,ximage,yimage                
                    data['MAG_APER'],data['MAGERR_APER'] = mag, magerr
                    data['FLAGS_EXTRACTION'], data['FLAGS_SCAMP'],data['FLAGS_IMA'] = flag1,flag2,flag3
                    #print('number chunks added this h5file:',len(idd))

                    c = astropy.table.Column(name='ifile', data=np.repeat(dither, len(idd))); 
                    data.add_column(c)
                    c = astropy.table.Column(name='idet' , data=np.repeat(idet, len(idd))); 
                    data.add_column(c)

                    #(x, y) = pixel2focalplane(data['idet'], data['X_IMAGE'], data['Y_IMAGE'])
                    c = astropy.table.Column(name='X_fp' , data=xfp); 
                    data.add_column(c)
                    c = astropy.table.Column(name='Y_fp' , data=yfp); 
                    data.add_column(c)

                    # Trasform x and y coordinates into the [-1,1] range on the detector
                    #(x, y) = pixel2det(idet, data['X_IMAGE'], data['Y_IMAGE'])
                    c = astropy.table.Column(name='X_det' , data=xdn); 
                    data.add_column(c)
                    c = astropy.table.Column(name='Y_det' , data=ydn); 
                    data.add_column(c)
       
                    #JXJXJX 10-25-2024 SpeedUp: Replaced the following section with the obs list.
                    '''
                    if len(obs) == 0:
                        obs = data.copy()
                    else:
                        obs = astropy.table.vstack([obs, data])
                    '''
                    if len(obs[n]) == 0:
                        obs[n] = data.copy()
                    else:
                        obs[n] = astropy.table.vstack([obs[n], data])                    
                #JXJXJX
        
        #JX 10-25-2024 SpeedUp: Moved the following section into a loop down below.
        '''
        nn = len(obs)
        # Sort by star magnitude
        i = np.argsort(obs['MAG_APER'])[::-1]
        obs = obs[i]
        print("Number of individual stars before repeat           : ", len(obs))

        ############ Added these two lines to remove non repeated observations and outlier magnitudes
        obs = self.remove_unique_object_ids(obs)
        print("Number of individual stars single ones removed           : ", len(obs))

        obs = self.sigmaclip(obs)
        print("Number of stars 3sigma clipped magnitudes           : ", len(obs))

        ################################################
        (catalog, index_star2obs, index_det2obs) = self.extractStarCatalog(obs)

        self.obs = obs
        self.catalog = catalog
        self.index_star2obs = index_star2obs
        self.index_det2obs = index_det2obs

        print("Number of coefficients in the model                : ", self.ncoeff)
        print("Number of individual stars                         : ", len(catalog))
        print("Number of constraints                              : ", len(obs))
 
        if self.outdir is not None:
            fig = plt.figure()
            plt.plot(obs['MAG_APER'], obs['MAGERR_APER'], marker='.', linestyle="None", ms=1.5)
            plt.xlabel("Magnitude")
            plt.ylabel("Mag. uncertainty")
            plt.yscale("log")
            plt.savefig(os.path.join(outdir, f'03-SourceMagAndUnc_w{wcenter}_g{grism_param}.png'), dpi=200)
        '''
        #JXJXJX
            
        #JX 10-25-2024 SpeedUp: Created the following lists and the loop on n.
        self.obs = [[] for _ in range(len(wcenter))] 
        self.catalog = [[] for _ in range(len(wcenter))]
        self.index_star2obs = [[] for _ in range(len(wcenter))]
        self.index_det2obs = [[] for _ in range(len(wcenter))]

        for n in range(len(wcenter)): 
            nn = len(obs[n])
            print("n=",n," wcenter_n=",wcenter[n])
            # Sort by star magnitude
            i = np.argsort(obs[n]['MAG_APER'])[::-1]
            obs[n] = obs[n][i]
            print("Number of individual stars before repeat           : ", len(obs[n]))

            ############ Added these two lines to remove non repeated observations and outlier magnitudes
            obs[n] = self.remove_unique_object_ids(obs[n])
            print("Number of individual stars single ones removed     : ", len(obs[n]))

            obs[n] = self.sigmaclip(obs[n])
            print("Number of stars 3sigma clipped magnitudes          : ", len(obs[n]))    

            ################################################
            (catalog[n], index_star2obs[n], index_det2obs[n]) = self.extractStarCatalog(obs[n])

            self.obs[n] = obs[n]
            self.catalog[n] = catalog[n]
            self.index_star2obs[n] = index_star2obs[n]
            self.index_det2obs[n] = index_det2obs[n]

            print("Number of coefficients in the model                : ", self.ncoeff)
            print("Number of individual stars                         : ", len(catalog[n]))
            print("Number of constraints                              : ", len(obs[n]))
 
            if self.outdir is not None:
                fig = plt.figure()
                plt.plot(obs[n]['MAG_APER'], obs[n]['MAGERR_APER'], marker='.', linestyle="None", ms=1.5)
                plt.xlabel("Magnitude")
                plt.ylabel("Mag. uncertainty")
                plt.yscale("log")
                #JX plt.savefig(os.path.join(outdir, "03-SourceMagAndUnc_w"+str(wcenter)+"_g"+str(grism_param)+".png"), dpi=200)
                plt.savefig(os.path.join(outdir, "03-SourceMagAndUnc_w"+str(wcenter[n])+"_g"+str(grism_param)+".png"), dpi=200)
        #JXJXJX
            
###################################################################################################
#################### SelfCalibration on wavelength bins [main]###########################################
###################################################################################################

class SelfCalibWave:
    """A class to run self calibration over wavelength direction"""
    
    def __init__(self, extspecnames, listallgrisms, ncoeff, limmag, calmode,smooth_kernel_sigma, noisefactor, deltaw, spatialres, waveres, ndither, mdb, optxml, outdir,ms_flat,tables=None):

        self.extspecnames = extspecnames
        self.listallgrisms = listallgrisms
        self.list_grisms = np.unique(listallgrisms) 
 
        self.tables = tables
        self.ncoeff = ncoeff
        self.limmag = limmag
        self.calmode = calmode
        self.noisefactor = noisefactor
        self.smooth_kernel_sigma = smooth_kernel_sigma
        self.deltaw = deltaw
        self.spatialres = spatialres
        
#        self.dq_parameters = ppr_dict.genericKeyValueParameters() #JX
        self.dq_parameters = {}
        
        self.waveres = waveres
        
        self.cube = np.zeros([self.waveres,self.spatialres,self.spatialres,6])# MakeCubeTess(ms_flat)
        self.errcube = np.zeros_like(self.cube)
        self.gap = np.zeros_like(self.cube)
        self.ms_flat = self.cube
        
        self.ndither = ndither
        self.outdir = outdir
        self.mdb = mdb
        self.optxml = optxml

    def fit(self):
        '''looping over wavelengths and filling in the cube using self calib fitting'''
                
        grisms_names = ['RGS000','RGS180+4','RGS000-4','RGS180','RGS270','BGS000'] #order of grisms!
        list_grisms_main =[0,6,-4,2,1,3] ### change 0 when blue grism and RGS270 data exists
        
        for ggbuf,grism in enumerate(self.list_grisms):
            print ("in SelfCalibWave fit(): ggbuf,grism= ",ggbuf,grism)
            if grism  == 3:
                #JX wstart = 11000.0 # 8548.5problematic
                wstart = 9000.0#8548.5
                wend = 13862.1
                wcenters = wstart+np.arange(1,self.waveres+1)*(wend-wstart)/(self.waveres+1)
            else:
                wstart = 11900.0
                wend = 19002.0
                wcenters = wstart+np.arange(1,self.waveres+1)*(wend-wstart)/(self.waveres+1)

            #JX 10-25-2024 SpeedUp: Moved the following section to outside of loop on i.
            '''   
            for i in range(self.waveres):
                spec_forGW = H5ExtractedSpectraCollection.load(self.extspecnames[0], ids=[H5ExtractedSpectrum.SPEC1D_LABEL,])     
                ## if self cal observations or simulations exist no need to dither things (default mode = 0)
                if self.calmode==0:
                    data2 = SelfCalibSirData(self.extspecnames,self.listallgrisms, self.ncoeff, self.limmag, wcenters[i],self.deltaw,grism,self.mdb,outdir=self.outdir)
                    
                ## simple random simulation of dithering spectra in the absence of SelfCal observations (mode = 1)    
                elif self.calmode==1:
                    data = SelfCalibSirData(self.extspecnames, self.ncoeff, self.limmag, wcenters[i],self.deltaw,grism,self.mdb,outdir=self.outdir)
                    data2 = SelfDitherData(data, self.mdb, spec_forGW.getGWAPositionName(), ndithers=self.ndither,noisefactor = self.noisefactor) 
                    
                ## More realistic simulation of selfCal pattern using GAIA star positions in the absence of SelfCal (mode=2)    
                else:
                    data2 = SelfCalibSirGaiaData(self.extspecnames,self.listallgrisms, self.tables, self.ncoeff, self.limmag, wcenters[i],self.deltaw,grism, self.noisefactor, self.ndither,self.mdb,outdir=self.outdir) 
                
                ##########debugging###################
                print('grism:',grisms_names[list_grisms_main.index(grism)])
                print('wave:',wcenters[i])
                ##########debugging###################
                m = SelfCalib(data2,grisms_names[list_grisms_main.index(grism)],self.mdb,self.optxml)
                m.fit()
                g = list_grisms_main.index(grism) 
                self.cube[i,:,:,g], self.errcube[i,:,:,g],self.gap[i,:,:,g] = m.OutputSir(self.spatialres,0)
                m.makePlots(wcenters[i])
                
                ### saving the catalog now for test/debug can be commented out
                ###np.savez('../star_catalog_g'+str(grism)+'_w'+str(wcenters[i])+'.npz',data2.obs,data2.catalog,data2.index_star2obs,data2.index_det2obs)
                '''
            #JX 10-25-2024 SpeedUp: Only load data once for each grism.
            spec_forGW = H5ExtractedSpectraCollection.load(self.extspecnames[0], ids=[H5ExtractedSpectrum.SPEC1D_LABEL,])     
            ## if self cal observations or simulations exist no need to dither things (default mode = 0)
            if self.calmode==0:
                data2 = SelfCalibSirData(self.extspecnames,self.listallgrisms, self.ncoeff, self.limmag, wcenters,self.deltaw,grism,self.mdb,outdir=self.outdir)

            ## simple random simulation of dithering spectra in the absence of SelfCal observations (mode = 1)    
            elif self.calmode==1:
                data = SelfCalibSirData(self.extspecnames, self.ncoeff, self.limmag, wcenters,self.deltaw,grism,self.mdb,outdir=self.outdir)
                data2 = SelfDitherData(data, self.mdb, spec_forGW.getGWAPositionName(), ndithers=self.ndither,noisefactor = self.noisefactor) 
                    
            ## More realistic simulation of selfCal pattern using GAIA star positions in the absence of SelfCal (mode=2)    
            else:
                data2 = SelfCalibSirGaiaData(self.extspecnames,self.listallgrisms, self.tables, self.ncoeff, self.limmag, wcenters,self.deltaw,grism, self.noisefactor, self.ndither,self.mdb,outdir=self.outdir) 
                
            #JX 10-25-2024 SpeedUp: Created the loop on i, and the Data2_i class.
            for i in range(self.waveres):
                ##########debugging###################
                print('grism:',grisms_names[list_grisms_main.index(grism)])
                #JX print('wave:',wcenters)
                print('wave:',wcenters[i])
                ##########debugging###################

                class Data2_i:
                    def __init__(self):
                        self.obs = ()
                        self.catalog = []
                        self.index_star2obs = []
                        self.index_det2obs = []
                data2_i = Data2_i()
                
                data2_i.obs = data2.obs[i] #JX Must copy, not append!
                data2_i.catalog = data2.catalog[i]
                data2_i.index_star2obs = data2.index_star2obs[i]
                data2_i.index_det2obs = data2.index_det2obs[i]
    
                data2_i.ncoeff = data2.ncoeff
                data2_i.limmag = data2.limmag
                data2_i.outdir = data2.outdir
                data2_i.wcenter = data2.wcenter
                data2_i.deltaw = data2.deltaw
                data2_i.mdb = data2.mdb

                m = SelfCalib(data2_i,grisms_names[list_grisms_main.index(grism)],self.mdb,self.optxml)
                m.fit()
                g = list_grisms_main.index(grism) 
                self.cube[i,:,:,g], self.errcube[i,:,:,g],self.gap[i,:,:,g] = m.OutputSir(self.spatialres,0)
                m.makePlots(wcenters[i])
                
                ### saving the catalog now for test/debug can be commented out
                ### np.savez('../star_catalog_g'+str(grism)+'_w'+str(wcenters[i])+'.npz',data2_i.obs,data2_i.catalog,data2_i.index_star2obs,data2_i.index_det2obs)
            #JXJXJX


    def cross_grism(self):
        ''' To correct for the grism to grism offset before saving the flat cube'''
        #for spectrum in refgrism_subsample:
        c = self.cube    
       
        X =  np.linspace(-1.,1.,np.shape(c)[1])
        Y =  np.linspace(-1.,1.,np.shape(c)[2])
        Z =  np.linspace(self.wstart,self.wend,np.shape(c)[0]) #JX

        fn000 = RGI((Z,X,Y), c[:,:,:,0])
        fn184 = RGI((Z,X,Y), c[:,:,:,1])
        fn004 = RGI((Z,X,Y), c[:,:,:,2])
        fn180 = RGI((Z,X,Y), c[:,:,:,3])
        fn270 = RGI((Z,X,Y), c[:,:,:,4])
        fn00B = RGI((Z,X,Y), c[:,:,:,5])
        fn = [fn000,fn184,fn004,fn180, fn270, fn00B]
     
    # need to improve later
        h5file1, h5file2, h5file3, h5file4, h5file5 = [],[],[],[],[]
        grispar = [0,6,-4,2]
        for k,h5file_name in enumerate(self.extspecnames):
            #grism_orientation = (h5file_name.split('/')[-1]).split('-')[2]

            grism_orientation = self.tables[k].getGWAPosition()+self.tables[k].getGWATilt()
            if (grism_orientation == grispar[0]) & (len(h5file1)==0): 
                h5file1 = H5ExtractedSpectraCollection.load(h5file_name, ids=[H5ExtractedSpectrum.SPEC1D_LABEL,])
            if (grism_orientation == grispar[1]) & (len(h5file2)==0): 
                h5file2 = H5ExtractedSpectraCollection.load(h5file_name, ids=[H5ExtractedSpectrum.SPEC1D_LABEL,])
            if (grism_orientation == grispar[2]) & (len(h5file3)==0): 
                h5file3 = H5ExtractedSpectraCollection.load(h5file_name, ids=[H5ExtractedSpectrum.SPEC1D_LABEL,])
            if (grism_orientation == grispar[3]) & (len(h5file4)==0): 
                h5file4 = H5ExtractedSpectraCollection.load(h5file_name, ids=[H5ExtractedSpectrum.SPEC1D_LABEL,])
 
        grisms = [h5file2,h5file3,h5file4]

        corrections = np.zeros((self.waveres,len(grisms)),dtype=np.ndarray)

        for i,spectrum in enumerate(h5file1):
            this_obj = spectrum.getObjectID()   
            detector = spectrum.getDetectorNumber()
            flux1 = spectrum.getSpectrumDataSet(H5ExtractedSpectrum.SPEC1D_LABEL).getActualData()
            var1 = spectrum.getSpectrumDataSet(H5ExtractedSpectrum.SPEC1D_LABEL).getActualVariance()
            quality1 = spectrum.getSpectrumDataSet(H5ExtractedSpectrum.SPEC1D_LABEL).getQuality()
            wave = spectrum.getWavelength()
            wstart = wave[0]
            wend = wave[-1]
            wcenters = wstart+np.arange(1,self.waveres+1)*(wend-wstart)/(self.waveres+1)

            m,dm = measureChunkMagnitude1D(wave,flux1,var1,quality1,wavelengths[1],500)                    
            if (m <18)&(dm<0.5):
                for g,grism in enumerate(grisms):
                    s2 = grism.getSpectraByObjectID(this_obj)
                    l,m2,dm2=0,99.0,99.0
                    for spectrum2 in s2:
                        l +=1
                        detector2 = spectrum2.getDetectorNumber()
                        flux2 = spectrum2.getSpectrumDataSet(H5ExtractedSpectrum.SPEC1D_LABEL).getActualData()
                        var2 = spectrum2.getSpectrumDataSet(H5ExtractedSpectrum.SPEC1D_LABEL).getActualVariance()
                        quality2 = spectrum2.getSpectrumDataSet(H5ExtractedSpectrum.SPEC1D_LABEL).getQuality()
                        wave2 = spectrum2.getWavelength()
                        m2,dm2 = measureChunkMagnitude1D(wave2,flux2,var2,quality2,wcenters[1],500)                    

                    if (l>0)&(m2<18)&(dm2<0.5):
                        for w,wavelength in enumerate(wcenters):        
                            x,y = spectrum.getLocationSpectrum().computePosition(wavelength, row=2).getPosition() # pixels
                            xd,yd = pixel2det(detector, x, y)  # detector
                            xf,yf = det2focalplane(detector, xd, yd) # focal plane
                            corfactor1 = 1
                            if (xf>-1) and (xf<1) and (yf>-1) and (yf<1):
                                pts = np.array([wavelength,xf,yf])    
                                corfactor1 = 10.0**(fn[0](pts)/-2.5)#compute_ff(c[:,:,:,0],xf,yf,wavelength) 
                    
                            x,y = spectrum2.getLocationSpectrum().computePosition(wavelength, row=2).getPosition() # pixels
                            xd,yd = pixel2det(detector2, x, y)  # detector
                            xf,yf = det2focalplane(detector2, xd, yd) # focal plane
                            corfactor2 = 1
                            if (xf>-1) and (xf<1) and (yf>-1) and (yf<1):
                                pts = np.array([wavelength,xf,yf])    
                                corfactor2 = 10.0**(fn[g+1](pts)/-2.5)#compute_ff(c[:,:,:,g+1],xf,yf,wavelength)   
                            
                            a = np.mean(flux1[find_nearest(wave,wavelength)-10:find_nearest(wave,wavelength)+10])/corfactor1
                            b = np.mean(flux2[find_nearest(wave2,wavelength)-10:find_nearest(wave2,wavelength)+10])/corfactor2
                
                            if (a/b)>0:
                                corrections[w,g] = np.append(corrections[w,g],-2.5*np.log10(a/b))

        
        grisms = ['RGS180+4','RGS000-4','RGS180']
        plt.figure(figsize=(12,12))
        knum=1
        for g,grism in enumerate(grisms):
            for w,wavelength in enumerate(wcenters):        
                self.cube[w,:,:,g+1]+=np.mean(stats.sigma_clip(corrections[w,g],sigma = 3))
            
                plt.subplot(len(grisms),len(wcenters),knum)
                no = plt.hist(stats.sigma_clip(corrections[w,g],sigma = 5),bins=20)
                plt.title('Grism:'+str(grism)+' at w:'+str(wavelength))
                plt.text(-1.5,10,'mean:'+str(np.mean(stats.sigma_clip(corrections[w,g],sigma = 5))))
                plt.xlabel("corrections in log")
                plt.ylabel("#")
                plt.plot([0,0],[0,30],'r--')
                plt.xlim([-2,2])
                knum+=1
        plt.savefig(os.path.join(self.outdir, "10-Crossgrism_corrections.png"), dpi=200)

                
    def quality_prod(self,sigmaclip = 10):
        ''' To check the quality of the calculated response'''
        
        #JX
        #Create an empty container
        print('Creating DQC...')
        
        ### fix grisms 
        response = self.cube    
        plt.figure(figsize=(20,20))
        list_grisms_main =[0,6,-4,2,1,3] 
        grisms = ['RGS000','RGS180+4','RGS000-4','RGS180','RGS270','BGS000']
        knum = 1

        for grism in list_grisms_main:#self.list_grisms:
            real_grism=grisms[list_grisms_main.index(grism)] 
            g = list_grisms_main.index(grism)
            avg_array=[] #JX
            std_array=[] #JX
            
            if grism  == 3:
                #JX wstart = 11000.0# 8548.5 problamatic all 0
                wstart = 9000.0#8548.5
                wend = 13862.1
                self.wcenters = wstart+np.arange(1,self.waveres+1)*(wend-wstart)/(self.waveres+1)
                wavelengths = self.wcenters
                
            else:
                wstart = 11900.0
                wend = 19002.0
                self.wcenters = wstart+np.arange(1,self.waveres+1)*(wend-wstart)/(self.waveres+1)
                wavelengths = self.wcenters


            for w,wavelength in enumerate(wavelengths):

                corrections = np.reshape(response[w,:,:,g],-1)
                ax = plt.subplot(len(list_grisms_main),len(wavelengths),knum)
                if len(corrections[np.isnan(corrections)])<len(corrections): #avoiding errors if all array is nan
                    no = plt.hist(stats.sigma_clip(corrections,sigma = sigmaclip),bins=20)
                    avg = np.mean(stats.sigma_clip(corrections[~np.isnan(corrections)],sigma = sigmaclip))
                    std = scipy.stats.median_abs_deviation(stats.sigma_clip(corrections[~np.isnan(corrections)],sigma = sigmaclip), scale='normal')
                    plt.text(0.12,0.9,'mean:'+str(np.round(avg,3)),horizontalalignment='center', verticalalignment='center',transform = ax.transAxes,fontsize=14)
                    plt.text(0.12,0.7,'NMAD:'+str(np.round(std,3)),horizontalalignment='center', verticalalignment='center',transform = ax.transAxes,fontsize=14)
                    plt.plot([avg,avg],[0,1000],'r--')
                    avg_array.append(avg) #JX
                    std_array.append(std) #JX
                    knum+=1

                plt.title('Grism:'+str(real_grism)+' at w:'+str(wavelength))
                plt.xlabel("corrections in log")
                plt.ylabel("#")
                plt.xlim([-0.5,0.5])

            name = 'Wavelengths'
            self.dq_parameters[name] = {'Key' : name,
                                        'Value' : np.round(wavelengths/10000.,2),
                                        'Description' : 'Wavelength (micron)',
                                        'Unit' : 'micron'}

            name = f'{real_grism}_avg'
            self.dq_parameters[name] = {'Key' : name,
                                        'Value' : np.array(avg_array),
                                        'Description' : f'{real_grism} mean correction (dmag)',
                                        'Unit' : 'dmag'}

            name = f'{real_grism}_std'
            self.dq_parameters[name] = {'Key' : name,
                                        'Value' : np.array(std_array),
                                        'Description' : f'{real_grism} stdDev (dmag)',
                                        'Unit' : 'dmag'}

            # param = ppr_dict.genericKVParam()
            # param.Key = f'Wavelengths'
            # param.Description = f'Wavelength (micron)'
            # param.StringValue = " ".join(np.array_str(np.round(wavelengths/10000.,2)).split()) #JX Convert wavelengths values to micron
            # param.Unit = ''
            # self.dq_parameters.Parameter.append(param)
            
            # param = ppr_dict.genericKVParam()
            # param.Key = f'{real_grism}_avg'
            # param.Description = f'{real_grism} mean correction (dmag)'
            # np_avg_arr = np.array(avg_array)
            # param.StringValue = " ".join(np.array_str(np_avg_arr).split())
            # #param.Unit = ''
            # self.dq_parameters.Parameter.append(param)

            # param = ppr_dict.genericKVParam()
            # param.Key = f'{real_grism}_std'
            # param.Description = f'{real_grism} stdDev (dmag)'
            # np_std_arr = np.array(std_array)
            # param.StringValue = " ".join(np.array_str(np_std_arr).split())
            # #param.Unit = ''
            # self.dq_parameters.Parameter.append(param)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.outdir, "11-quality_histograms.png"), dpi=200)
        

    #JX
    def get_dq_parameters(self) -> dict:
        """Returns the current dqc parameters

        Returns:
            dict: dqc parameters dict ready to be written in XML objet
        """
        return self.dq_parameters
        
        
    def writeOutputSir(self, outfile, msfits, iterative):
        mm = self
        
        if os.path.isfile(outfile):
            os.remove(outfile)
        print("Saving output file in ", outfile)
        
        hdul = fits.HDUList([fits.PrimaryHDU()])
        ts = time.time() 
        hdul[0].header['TELESCOP']=             'Euclid'
        hdul[0].header['INSTRUME']=               'NISP'
        hdul[0].header['FITS_DEF']=      'sir.relativeFuxScaling'
        hdul[0].header['FITS_VER']=                '0.1'
        hdul[0].header['VERSION'] =     'SC8-NIP-F3_159'
        hdul[0].header['SOFTVERS'] =     0.0     
        hdul[0].header['DATE']    =      datetime.utcfromtimestamp(ts).strftime('%Y-%m-%dT%H:%M:%S')
        hdul[0].header['OBSMODE'] =            'CALIBRATION'
        hdul[0].header['ORIGIN']  =               'Shooby'
        hdul[0].header['IMG_CAT'] =              'CALIB'
        hdul[0].header['ValidityRange'] = '2050-01-01T00:00:00.0'
        hdul[0].header['CalibrationID'] = '1'
        
        grisms = ['RGS000','RGS180+4','RGS000-4','RGS180', 'RGS270', 'BGS000']
        pos = ['RGS000','RGS180','RGS000','RGS180','RGS270','BGS000']
        tilt =[0,4,-4,0,0,0]
        
        for i,g in enumerate(grisms):
            print("Grism/Tilt ", g)
            npix = mm.spatialres

            header = fits.Header()
            header["EXTNAME"]= g 
            header["GWA_POS"]= pos[i]
            header["GWA_TILT"]= tilt[i]
                        
            #X-axis (dispersion) on focal plane
            header["CTYPE1"]= 'FP'
            header["CUNIT1"]= 'norm'
            header["CRVAL1"]= -0.99
            header["CRPIX1"]= 1.
            header["CDELT1"]= 2./(npix)

            #Y-axis (spatial) on focal plane
            header["CTYPE2"]= 'FP'
            header["CUNIT2"]= 'norm'
            header["CRVAL2"]= -0.99
            header["CRPIX2"]= 1.
            header["CDELT2"]= 2./(npix)

            if g  == 'BGS000':
                wstart = 9000.0#8548.5
                wend = 13862.1
                wcent = wstart+np.arange(1,self.waveres+1)*(wend-wstart)/(self.waveres+1)
            else:
                wstart = 11900.0
                wend = 19002.0
                wcent = wstart+np.arange(1,self.waveres+1)*(wend-wstart)/(self.waveres+1)

            #Wavelength dimension
            header["CTYPE3"]= 'Wave'
            header["CUNIT3"]= 'nm'
            header["CRVAL3"]= wcent[0] #goes to self.wvcent[0]
            header["CRPIX3"]= 1.
            if len(wcent) > 1:
                header["CDELT3"]= wcent[1]-wcent[0]
            else:
                header["CDELT3"]= 0. 
                        
                    
            
            if (self.smooth_kernel_sigma>0) & (np.sum(mm.cube[0,:,:,i])!=0):
                cu_wave = sigma_clipped_mean(mm.cube[:,:,:,i], sigma=1) #remove outlier in wavelength direction
                cu_spatial = smooth_avoidgaps(cu_wave, mm.gap[self.waveres//2,:,:,i], sigma=self.smooth_kernel_sigma) # smooth spatially
                cu = np.tile(cu_spatial[np.newaxis, :, :], (self.waveres, 1, 1)) # put the same smooth solution in all wavebins
                print(g,'done----------')
            else:
                cu = mm.cube[:,:,:,i] # no smoothing in w or spatially

            cu_smooth = nan_helper(cu,sigma=0)
            cube_swap = np.moveaxis(cu_smooth,2,1) #swap axis for consistency
            
            if iterative:   
                incube,inerrcube = MakeCubeTess(msfits) # this is the input ms_flat file from previous run
                hdu = fits.ImageHDU(data=cube_swap+incube[:,:,:,i], header=header, name= g + '.SCI') #iterative
            else:
                hdu = fits.ImageHDU(data=cube_swap, header=header, name= g + '.SCI')

            hdul.append(hdu)

            #The DQ layer will be a sigmoid function of S/N of the solution.
            #For S/N ~ 5 and above it goes to 1, for S/N ~ 1 and below it goes to 0. -DCM
            
            DQ_layer = np.ones_like(mm.cube[:,:,:,i])
            
            for j in range(self.waveres):

                mmcubeprime = np.copy(mm.cube[j,:,:,i])
                mmcubeprime[np.where(np.isnan(mmcubeprime))] = 0.0

                mmerrcubeprime = np.copy(mm.errcube[j,:,:,i])
                mmerrcubeprime[np.where(np.isnan(mmerrcubeprime))] = 10.0
                
                xvals = -120.*mmerrcubeprime+5.4    
                DQ_layer[j,:,:] = 1 / (1 + np.exp(-xvals))
            
            dq_smooth = nan_helper(DQ_layer,sigma=1)
            dq_swap = np.moveaxis(dq_smooth,2,1) #swap axis for consistency

            hdu = fits.ImageHDU(data=dq_swap, header=header, name=g + '.DQ')
            hdul.append(hdu)

        hdul.writeto(outfile)

        
###################################################################################################
#################### per grism per waveband fitting similar to NIR#################################
###################################################################################################

class SelfCalib:
    """A class to implement Self-calibration algorithms"""

    def __init__(self, data, grismname, mdb,optxml):
        self.data = data
        self.par_stars = data.catalog.copy()
        self.par_detqe = np.zeros(16)
        par_flat = []
        for idet in range(0, 16):
            i = data.index_det2obs[idet]
            x = data.obs['X_det'][i]
            y = data.obs['Y_det'][i]
            par_flat.append(Fit2DNP(data.ncoeff, x, y))
            #par_flat.append(Fit2DCheb(data.ncoeff, x, y))
        self.par_flat = np.asarray(par_flat)
        self.mdb = mdb
        self.optxml = optxml
        self.grismname = grismname
        
    def expectedUncertainties(self):
        m = self
        return np.mean(m.data.obs['MAGERR_APER']) / np.sqrt(len(m.data.obs) / (m.data.ncoeff**2 * 16))

    def obsData(self):
        return self.data.obs['MAG_APER']

    def obsUncert(self):
        return self.data.obs['MAGERR_APER']

    def model_starMag(self):
        return self.par_stars['mag'][self.data.obs['istar']]

    def model_flat(self):
        m = self
        out = np.zeros(len(m.data.obs))
        for idet in range(0, 16):
            i = m.data.index_det2obs[idet]
            out[i] = m.par_flat[idet].predict()
        return out

    def model_detQE(self):
        return self.par_detqe[self.data.obs['idet']-1]

    def model(self):
        return self.model_starMag() + self.model_flat() + self.model_detQE()

    def residuals(self):
        return (self.obsData() - self.model()) / self.obsUncert()

    def chisquare(self):
        return np.sum(self.residuals()**2)

    def redchisquare(self):
        npar_flat = 16 * self.par_flat[0].param_val.size
        dof = len(self.data.obs) - len(self.par_stars) - npar_flat - self.par_detqe.size
        return self.chisquare() / dof

    def fitFlat(self):
        m = self
        d = m.obsData() - (m.model_starMag() + m.model_detQE())
        for idet in range(0, 16):
            i = m.data.index_det2obs[idet]
            m.par_flat[idet].fit(d[i], m.obsUncert()[i])

    def fitStarMag(self):
        m = self
        unc = m.obsUncert()
        d = m.obsData() - (m.model_flat() + m.model_detQE())
        #for i in range(0, len(m.par_stars)):
        #    j = m.data.index_star2obs[i]
        #    mask_mag = np.ma.MaskedArray(d[j], mask=np.isnan(d[j]))
        #    mask_err = np.ma.MaskedArray(unc[j], mask=np.isnan(d[j]))
        #    m.par_stars['mag'][i] = np.ma.average(mask_mag, weights=1./mask_err)
        for i in range(0, len(m.par_stars)):
            j = m.data.index_star2obs[i]
            m.par_stars['mag'][i] = np.average(d[j], weights=1./unc[j])
            m.par_stars['mag'][i] = np.mean(d[j])

    def flattenResiduals(self):
        bkp = self.par_detqe.copy()
        chi0 = self.redchisquare()
        d = self.obsData() - self.model()
        for i in range(0, 16):
            j = self.data.index_det2obs[i]
            self.par_detqe[i] += np.median(d[j])
        chi1 = self.redchisquare()
        if (chi1 - chi0) > 0:
            self.par_detqe[:] = bkp
        return chi1 - chi0

    def fitQE(self):
        m = self
        mat = np.zeros(16, dtype='int')
        data = []

        # fig, ax = plt.subplots()
        # ax = fig.add_subplot(111, projection='3d')
        for det1 in NIRFP().detnames():
            for det2 in NIRFP().detnames():
                if det_neighbour(det1, det2):
                    if int(det1) > int(det2):
                        continue  # avoid repetitions
                    fp = FPCoordsGaps(det1, det2, 5);
                    cc1 = fp.toDetCoords(use_det=det1)
                    cc2 = fp.toDetCoords(use_det=det2)
                    idet1 = DetectorModel.getDetectorNumber(det1)
                    idet2 = DetectorModel.getDetectorNumber(det2)
                    

                    idet1_maybe = (int(det1[0])-1) * 4 + (int(det1[1])-1)
                    idet2_maybe = (int(det2[0])-1) * 4 + (int(det2[1])-1)

                    for i in range(0, fp.x.size):
                        v1 = m.par_flat[idet1_maybe].extrapolate(cc1.x[i], cc1.y[i])
                        v2 = m.par_flat[idet2_maybe].extrapolate(cc2.x[i], cc2.y[i])
                        # ax.scatter(fp.x[i], fp.y[i], v1, marker='.', s=np.abs(v1) * 100)
                        # ax.scatter(fp.x[i], fp.y[i], v2, marker='.', s=np.abs(v2) * 100)
                        data.append(v1 - v2)
                        row = np.zeros(16, dtype='int')
                        row[(int(det1[0])-1) * 4 + (int(det1[1])-1)] =  1
                        row[(int(det2[0])-1) * 4 + (int(det2[1])-1)] = -1
                        mat = np.vstack((mat, row))
        data = np.asarray(data)

        # Add constraint for the average and drop first row from mat
        data = np.insert(data, data.size, 0.)
        mat = np.vstack((mat[1:,:], np.repeat(1, 16)))
        res = np.linalg.lstsq(mat, data, rcond=-1.)[0]

        for det in NIRFP().detnames():
            idet = DetectorModel.getDetectorNumber(det)
            i = (int(det[0])-1) * 4 + (int(det[1])-1)
            m.par_detqe[i] += res[i]
            m.par_flat[i].addGlobalOffset(-1. * res[i])

        
        out = np.sum(np.abs(data))
        return out


    def fit(self):
        m = self
        elapsed = time.time()
        print("Expected uncertainties (at each location): ", m.expectedUncertainties())
        print("Initial chisq. ", m.redchisquare())
        iter = 0
        while True:
            iter += 1
            chi0 = m.redchisquare()
            m.fitStarMag()
            m.flattenResiduals()  # ensure residuals are flattened across the detectors
            m.fitFlat()
            chi1 = m.redchisquare()
            deltachi = chi1 - chi0
            s = "Iter {iter:4d},  red. chisq.: {chi1:10.6g}   delta: {deltachi:10.6g}".format(iter=iter, chi1=chi1, deltachi=deltachi)
            print(s)
            if deltachi > 0 or iter>100:
                for i in range(0, 16):
                    m.par_flat[i].restore()
                m.fitStarMag()
                break
        print("Final chisq. ", m.redchisquare())
        printStatistics("Residuals" , m.residuals())
        printStatistics("Flat [mag] (pre-QE) ", m.model_flat())
        #m.fitQE()  # Flatten large scale pattern across all detectors (chi sq. is unaffected)
        printStatistics("QE [mag]"  , m.par_detqe)
        printStatistics("Flat [mag] (post-QE)", m.model_flat())
        elapsed = time.time() - elapsed
        print("Elapsed time: ", elapsed, " [s]")


    def interpolateOnDetector(self, _idet, x, y):
        m = self
        idet = _idet - 1

        (_x, _y) = cartesianProduct(x, y)
        val = m.par_flat[idet].predict_at(_x, _y)
        unc = m.par_flat[idet].uncert_at( _x, _y)
        val = val.reshape(x.size, y.size)
        unc = unc.reshape(x.size, y.size)

        val = val.astype(np.float32)
        unc = unc.astype(np.float32)
        return (val, unc)
    
    def interpolateOnFocal(self, x, y):
        m = self

        (_x, _y) = cartesianProduct(x,y)
        
        #######latest by shoob Aug 11,2023##############
        detmod = DetectorModel(self.grismname, self.mdb, rotate=True)
        opt=OpticalModel.load('.', self.optxml, detmod)
        box = detmod.getEnvelopeBox()
        
        _nx,_ny= normalfov_r(box,(_x,_y))
        
        v,u = [],[]
        gap = []
        for d in range(len(_nx)):
            p= detmod.getPixel(_nx[d],_ny[d])      
            detx,dety =normaldet((p[0],p[1]))
            idet = p.getDetectorNumber()-1
            if idet==-1:
                v.append(0) ## change to m.par_flar
                u.append(m.par_flat[idet].uncert_at( detx, dety)) ## same
                gap.append(-1)
            else:
                v.append(m.par_flat[idet].predict_at(detx, dety)) ## change to m.par_flar
                u.append(m.par_flat[idet].uncert_at( detx, dety)) ## same
                gap.append(idet)
            #print(idet,v)
        v = np.array(v)
        u = np.array(u)
        gap = np.array(gap)
        
        val = v.reshape(x.size, y.size)
        unc = u.reshape(x.size, y.size)
        gap = gap.reshape(x.size, y.size)
        
        val = val.astype(np.float32)
        unc = unc.astype(np.float32)
        gap = gap.astype(np.float32)
        
        return (val, unc, gap)
    
    def smoothbaba(self,a,gap,sigma=2):

        copya=a
        for idet in range(0, 16):
            minx,maxx,miny,maxy = find_bounds(gap,idet-1)
            im = a[minx:maxx,miny:maxy]
            copya[minx:maxx,miny:maxy] = gaussian_filter(im,sigma=sigma)
        return copya
    
    def OutputSir(self, xsize,smooth_kernel_sigma):
        m = self
        cube = np.zeros([xsize,xsize])
        pix_x = np.linspace(-1.0, 1.0, xsize)
        pix_y = pix_x.copy()
        (val, unc, gap) = m.interpolateOnFocal(pix_x, pix_y)
        if smooth_kernel_sigma>0:
            val = m.smoothbaba(val,gap,sigma=smooth_kernel_sigma)
        return (val,unc,gap)


    def makePlots(self,wave):
        m = self
        outdir = m.data.outdir
        mdb = self.mdb
        #box=[[-82.5,-78],[82.5,78]] ### Focal plane in mm
        # Residuals
    
        detmod = DetectorModel(self.grismname, mdb, rotate=True)
        opt=OpticalModel.load('.',self.optxml, detmod)
        box = detmod.getEnvelopeBox()

        # Residuals
    
        fig = plt.figure()
    
        plt.xlabel("Obs. constraint (color=detector)")    
        plt.ylabel("Norm. residuals")
        i = range(0, len(m.data.obs))
        j = np.argsort(m.data.obs['idet'])
        tmp = m.residuals()
        plt.scatter(i, tmp[j], marker="s", s=np.abs(tmp[j]/5), c=m.data.obs['idet'][j], cmap=plt.get_cmap('brg'))
        plt.plot(i, np.repeat( 0, len(i)), color='gray')
        plt.plot(i, np.repeat( 1, len(i)), color='gray')
        plt.plot(i, np.repeat(-1, len(i)), color='gray')
        plt.ylim([-10,10])
        count = 0
        for idet in range(0, 16):
            j = m.data.index_det2obs[idet]
            tmp = m.residuals()[j]
            plt.plot(np.repeat(count+len(j), 2), [-5,5], color='gray')
            plt.scatter( count + len(j)/2, np.median(tmp), marker='x', color='black')
            plt.errorbar(count + len(j)/2, np.mean(tmp), yerr=np.std(tmp), marker='+', color='black')
            count += len(j)
        plt.savefig(os.path.join(outdir, "04-Residuals_w"+str(np.round(wave,0))+'_'+str(self.grismname)+".png"), dpi=200)

        fig = plt.figure()
        plt.xlabel("Norm. residuals")  # TODO: check this plot...
        plt.ylabel("Counts")
        tmp = m.residuals().copy()
        n, bins, patches = plt.hist(tmp, 100, (-5, 5), facecolor='blue', alpha=0.5)
        norm = np.sum(n) * (bins[1]-bins[0])
        plt.plot(bins, norm * gaussian(bins, 0., 1.0), color='red')
        plt.savefig(os.path.join(outdir, "05-ResidualsHisto_w"+str(np.round(wave,0))+'_'+str(self.grismname)+".png"), dpi=200)



        fig = plt.figure()
        tmp = m.residuals().copy()
        i = np.where(np.abs(tmp) > 5)
        tmp[i] = 5 * np.sign(tmp[i])

        if len(tmp[np.isnan(tmp)])<len(tmp):
            plt.title("Norm. residuals")
            plt.xlabel("Focal plane Y")
            plt.ylabel("Focal plane Z")
            plt.xlim([-1,1]); plt.ylim([1,-1])
            plt.scatter(m.data.obs['X_fp'], m.data.obs['Y_fp'], c=tmp, marker="s", s=np.abs(tmp/5), cmap=plt.get_cmap('bwr')) 
            plt.colorbar()
            plt.savefig(os.path.join(outdir, "06-ResidualsOnFocalPlane_w"+str(np.round(wave,0))+'_'+str(self.grismname)+".png"), dpi=200)


        fig, ax = plt.subplots()
        ax = fig.add_subplot(111, projection='3d')
        x = np.linspace(-1., 1., m.data.ncoeff+1); x = (x[0:-1] + x[1:]) / 2.
        y = x.copy()
        xdet_p,ydet_p = normaldet_r((x,y)) # in pixel on detector

        for idet in DetectorModel.getDetectorRange():
            det_id = DetectorModel.getDetectorID(idet)
            (val, unc) = m.interpolateOnDetector(idet, x, y)
            _x,_y=[],[]
            for xdp,ydp in zip(xdet_p,ydet_p):
                e = detmod.getFOVPosition(xdp,ydp,det_id)
                _x.append(e[0])
                _y.append(e[1])
            _x,_y = np.array(_x),np.array(_y) # on focal plane in mm
            nx,ny= normalfov(box,(_x,_y)) #on focalplane normal
            _nx,_ny = np.meshgrid(nx,ny)

            ax.plot_surface(_ny, _nx, val)
        plt.title("Flat field pattern")
        plt.xlabel("Focal plane Y")
        plt.ylabel("Focal plane Z")
        #plt.xlim([-1,1]); plt.ylim([1,-1])
        plt.savefig(os.path.join(outdir, "07-FlatPattern3D_w"+str(np.round(wave,0))+'_'+str(self.grismname)+".png"), dpi=200)

        fig, ax = plt.subplots()
        rr = np.zeros(2)
        x = np.linspace(-1., 1., m.data.ncoeff+1); x = (x[0:-1] + x[1:]) / 2.
        y = x.copy()
        xdet_p,ydet_p = normaldet_r((x,y)) # in pixel on detector

        for idet in DetectorModel.getDetectorRange():
            (val, unc) = m.interpolateOnDetector(idet, x, y)
            rr[0] = -0.2#np.min((rr[0], val.min()))
            rr[1] = 0.2#np.max((rr[1], val.max()))
        for idet in DetectorModel.getDetectorRange():
            det_id = DetectorModel.getDetectorID(idet)
            (val, unc) = m.interpolateOnDetector(idet, x, y)
            _x,_y=[],[]
            for xdp,ydp in zip(xdet_p,ydet_p):
                e = detmod.getFOVPosition(xdp,ydp,det_id)
                _x.append(e[0])
                _y.append(e[1])
            _x,_y = np.array(_x),np.array(_y) # on focal plane in mm
            nx,ny= normalfov(box,(_x,_y)) #on focalplane normal
            _x = boundaries(nx)
            _y = boundaries(ny)
            im = ax.pcolor(_y, _x, val, vmin=rr[0], vmax=rr[1], cmap=plt.get_cmap('brg'))
        cbar = fig.colorbar(im)
        plt.title("Flat field pattern")
        plt.xlabel("Focal plane Y")
        plt.ylabel("Focal plane Z")
        plt.xlim([-1,1]); plt.ylim([1,-1])
        plt.savefig(os.path.join(outdir, "07-FlatPattern_w"+str(np.round(wave,0))+'_'+str(self.grismname)+".png"), dpi=200)


        fig, ax = plt.subplots()
        tmp = m.data.obs['MAG_APER'] - m.par_stars['mag'][m.data.obs['istar']] - m.par_detqe[m.data.obs['idet']-1]

        if len(tmp[np.isnan(tmp)])<len(tmp):

            sc = ax.scatter(m.data.obs['X_fp'], m.data.obs['Y_fp'], vmin=tmp.mean()-2*tmp.std(), vmax=tmp.mean()+2*tmp.std(), \
                   marker='.', c=tmp, s=0.01, cmap=plt.get_cmap('brg'))
            fig.colorbar(sc,ax=ax)
        plt.title("Flat field pattern")
        plt.xlabel("Focal plane Y")
        plt.ylabel("Focal plane Z")
        plt.xlim([-1,1]); plt.ylim([1,-1])
        plt.savefig(os.path.join(outdir, "07-FlatPatternObs_w"+str(np.round(wave,0))+'_'+str(self.grismname)+".png"), dpi=200)

        fig, ax = plt.subplots()
        tmp = m.par_detqe[m.data.obs['idet']-1]

        if len(tmp[np.isnan(tmp)])<len(tmp):
            sc = ax.scatter(m.data.obs['X_fp'], m.data.obs['Y_fp'], vmin=-1, vmax=1, marker='.', c=tmp, s=0.01, cmap=plt.get_cmap('brg'))
            fig.colorbar(sc,ax=ax)
        plt.title("QE")
        plt.xlabel("Focal plane Y")
        plt.ylabel("Focal plane Z")
        plt.xlim([-1,1]); plt.ylim([1,-1])
        plt.savefig(os.path.join(outdir, "07-detq_w"+str(np.round(wave,0))+'_'+str(self.grismname)+".png"), dpi=200)


        fig, ax = plt.subplots()
        rr = np.zeros(2)
        x = np.linspace(-1., 1., m.data.ncoeff+1); x = (x[0:-1] + x[1:]) / 2.
        y = x.copy()
        xdet_p,ydet_p = normaldet_r((x,y)) # in pixel on detector

        for idet in DetectorModel.getDetectorRange():
            (val, unc) = m.interpolateOnDetector(idet, x, y)
            rr[0] = np.min((rr[0], unc.min()))
            rr[1] = np.max((rr[1], unc.max()))
        for idet in DetectorModel.getDetectorRange():
            det_id = DetectorModel.getDetectorID(idet)  
            
            _x,_y=[],[]
            for xdp,ydp in zip(xdet_p,ydet_p):
                e = detmod.getFOVPosition(xdp,ydp,det_id)
                _x.append(e[0])
                _y.append(e[1])
            _x,_y = np.array(_x),np.array(_y) # on focal plane in mm
            nx,ny= normalfov(box,(_x,_y)) #on focalplane normal

            (val, unc) = m.interpolateOnDetector(idet, x, y)
            _x = boundaries(nx)
            _y = boundaries(ny)
            im = ax.pcolor(_y, _x, unc, vmin=rr[0], vmax=rr[1], cmap=plt.get_cmap('brg'))
        plt.title("Flat field uncertainties")
        plt.xlabel("Focal plane Y")
        plt.ylabel("Focal plane Z")
        plt.xlim([-1,1]); plt.ylim([1,-1])
        plt.savefig(os.path.join(outdir, "08-FlatUncert_w"+str(np.round(wave,0))+'_'+str(self.grismname)+".png"), dpi=200)


        x = np.linspace(-1., 1., m.data.ncoeff+1); x = (x[0:-1] + x[1:]) / 2.
        y = x.copy()
        allv = np.empty((0))
        allu = np.empty((0))
        for idet in DetectorModel.getDetectorRange():
            (val, unc) = m.interpolateOnDetector(idet, x, y)
            allv = np.hstack([allv, val.reshape(val.size)])
            allu = np.hstack([allu, unc.reshape(unc.size)])
        

        if len(allv[np.isnan(allv)])<len(allv):
            fig = plt.figure()
            n, bins, patches = plt.hist(allv, bins=30)
            plt.xlabel("Delta magnitude")
            plt.savefig(os.path.join(outdir, "09-FlatPatternHisto_"+str(wave)+'_'+str(self.grismname)+".png"), dpi=200)

            fig = plt.figure()
            n, bins, patches = plt.hist(allu, range=[allu.mean()-4*allu.std(), allu.mean()+4*allu.std()], bins=50)
            plt.plot(np.array((0.006, 0.006)), np.array((0, n.max())), color='red', linestyle='--')
            plt.xlabel("Uncertainties [mag]")
            plt.savefig(os.path.join(outdir, "09-FlatUncertHisto_w"+str(np.round(wave,0))+'_'+str(self.grismname)+".png"), dpi=200)
        
        plt.close('all')
