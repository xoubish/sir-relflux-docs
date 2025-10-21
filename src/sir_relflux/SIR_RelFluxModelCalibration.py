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

#File: python/SIR_SelfCalib/SIR_Compute.py
#Created on: 04/14/20
#Author: shemmati, Georgio
#Compute Self-calibration image using the relative flux estimates of wavelength chunks of spectra taken from the decontaminated spectra.
#
#Example:
#ERun SIR_Calibration SIR_RelFluxModelCalibration \
#  --workdir=/stage/euclid-staff-shemmati/Work/Projects/spectra/R13 
#  --inlist=R13_specs_xmls.json 
#  --ms_flat=SIR_RelativeFluxScaling.xml
# --mdbfile=EUC_MDB.xml

import os
import json
import argparse
import ElementsKernel.Logging as log
#import xml.etree.ElementTree as ET

from SIR_Products.DpdMdbDataBase import DpdMdbDataBase
from SIR_Products.DpdSirRelativeFluxScaling import DpdSirRelativeFluxScaling
from SIR_Products.DpdSirExtractedSpectraCollection import DpdSirExtractedSpectraCollection

#from SIR_SpectraExtraction.DpdExtractedSpectraCollection import DpdExtractedSpectraCollection
#from ST_DataModelBindings.dpd.sir import extractedspectracollection_stub

import matplotlib
matplotlib.use('Agg')

from SIR_FluxCalibration.SelfCalib import *

#from ST_DataModelBindings.dpd.sir import locationtable_stub
#from ST_DataModelBindings.dpd.sir import relativeflux_stub
#from ST_DataModelBindings.pro import sir_stub
#from ST_DataModelBindings.sys import dss_stub
#import ST_DataModelBindings.ins_stub as ins_stub
#from ST_DM_FilenameProvider.FilenameProvider import FileNameProvider
#from SIR_Utilities.DpdGeneric import DpdGeneric
#from SIR_Utilities.Mdb import Mdb


# class DpdRelFluxScaling(DpdGeneric):
#     @classmethod
#     def make_header(cls):
#         return cls._createSirHeader('DpdSirRelativeFluxScaling')

def get_input_data(workdir, extr_list):
    """ get input XML file name """
    _, ext = os.path.splitext(extr_list)

    if ext == ".xml":
        return extr_list

    if ext == ".json":
        with open(os.path.join(workdir,extr_list),'r') as f:
            list_file = json.load(f)
        if len(list_file) == 0:
            return None

        return list_file

    raise Exception(f"Invalid input file {extr_list}")

# def getGWTiltxml(workdir,xml_path):
#     spectra_xml_obj = DpdSirExtractedSpectraCollection(os.path.join(workdir,xml_path))
#     return spectra_xml_obj.get_gwa_tilt()
#     # with open(os.path.join(workdir,xml_path)) as xml_file:
#     #     xml_text = xml_file.read()
#     #     dpd = extractedspectracollection_stub.CreateFromDocument(xml_text)
#     # return dpd.Data.GrismWheelTilt

# def getGWPosxml(workdir,xml_path):
#     spectra_xml_obj = DpdSirExtractedSpectraCollection(os.path.join(workdir,xml_path))
#     return spectra_xml_obj.get_gwa_pos()
#     # with open(os.path.join(workdir,xml_path)) as xml_file:
#     #     xml_text = xml_file.read()
#     #     dpd = extractedspectracollection_stub.CreateFromDocument(xml_text)
#     # return dpd.Data.GrismWheelPos

def defineSpecificProgramOptions():
    """
    @brief Allows to define the (command line and configuration file) options
    specific to this program

    @details
        See the Elements documentation for more details.
    @return
        An  ArgumentParser.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--workdir', type=str, dest='workdir',
                        help='Working directory path', required=True)
    parser.add_argument('--inlist', type=str, dest='inlist',
                        help='Input file name containing the list of extracted 1D spectra (JSON)', required=True)
    parser.add_argument('--inloclist', type=str, dest='inloclist',
                        help='Input file name containing the list of XML Location tables (JSON)', required=False)
    parser.add_argument('--ncoeff', type=str, dest='ncoeff', default=16,
                        help='Number of model coefficients per side of detector (default: 16)')
    parser.add_argument('--limmag', type=float, dest='limmag', default=18,
                        help='Limiting H magnitude of star to be used for calibration (default: 18)')
    parser.add_argument('--ms_flat', type=str,default='SIR_RelativeFluxScaling.xml',
                        help='The starting flat field solution product')
    parser.add_argument('--calmode', type=int, dest='calmode', default=0,
                        help='0 for SelfCal data, 1 for random dither positions, 2 for GAIA matched stars (default:0)')
    parser.add_argument('--iterative', type=int, dest='iterative', default=0,
                        help='0 for no iteration, 1 for adding with input response(default:1)')
    parser.add_argument('--smooth_kernel_sigma', type=int, dest='smooth_kernel_sigma', default=3,
                        help='width of the gaussian kernel to smooth per detector responses in the output array')
    parser.add_argument('--mdbfile', type=str, required=True,
                        help='The MDB file')
    parser.add_argument('--optxmlfile', type=str, default='opt.xml',
                        help='The OPT file')
    parser.add_argument('--output_xml', type=str, dest='output_xml',help='The output xml file', default='SIR_RelativeFluxScaling.xml') #JX
    parser.add_argument('--logdir', type=str, required=False,
                        help='The log directory')

    return parser


def mainMethod(args):
    """
    @brief The "main" method.
    @details
        This method is the entry point to the program. In this sense, it is
        similar to a main (and it is why it is called mainMethod()).
    """

    logger = log.getLogger('SIR_RelFluxModelCalibration')
    logger.info('#')
    logger.info('# Entering SIR_RelFluxModelCalibration mainMethod()')
    logger.info('#')

    if not os.path.isdir(args.workdir):
        msg = 'Work dir {} not found'.format(args.workdir)
        logger.info(msg)
        raise IOError(msg)

    infile = os.path.join(args.workdir, args.inlist)
    if not os.path.isfile(infile):
        msg = 'Input JSON file {} not found'.format(infile)
        logger.info(msg)
        raise IOError(msg)


    #inlocfile = os.path.join(args.workdir, args.inloclist)
    #if not os.path.isfile(inlocfile):
    #    msg = 'Input JSON file {} not found'.format(inlocfile)
    #    logger.info(msg)
    #    raise IOError(msg)


    # Log validated command line arguments
    logger.info(f'Input JSON file: {infile}')
    logger.info(f'       work dir: {args.workdir}')
    grismdict = {'RGS000':0,'RGS180':2,'BGS000':3,'RGS270':1} # not sure about 270 no data to check
    in_xml = get_input_data(args.workdir, infile)
    extspecnames=[]
    list_grisms = []
    for xmlname in in_xml:
        # get HDF5 filename from XML
        # print(args.workdir,xmlname)
        # extspecnames.append(DpdExtractedSpectraCollection.getHDF5FileNames(args.workdir, xmlname)[0])
        # list_grisms.append(grismdict[getGWPosxml(args.workdir, xmlname)]+getGWTiltxml(args.workdir, xmlname))
        spectra_xml_obj = DpdSirExtractedSpectraCollection(os.path.join(args.workdir, xmlname))
        extspecnames.append(os.path.join('data',spectra_xml_obj.get_hdf5_filename()))
        list_grisms.append(grismdict[spectra_xml_obj.get_gwa_pos()] + spectra_xml_obj.get_gwa_tilt())
    
    ######## change on march152024
    list_grisms = np.array(list_grisms)#np.unique(np.array(list_grisms)) # this identifies which grisms will be in the calibration         
    # mdb = Mdb.load(args.workdir, args.mdbfile)
    mdb = DpdMdbDataBase(args.mdbfile, workdir=args.workdir)

    optxml = os.path.join(args.workdir, args.optxmlfile)

    

    ## need this NISP detector slots for some NIR detector layout routines##       
    # csv = os.path.join(args.workdir, 'data', mdb.get('SpaceSegment.Instrument.NISP.NISPDetectorSlots')[0])
    csv = os.path.join(args.workdir, 'data', mdb.get_nisp_detector_slots_file())
    logger.info('        WCS CSV: ' + csv)
    NIRFP.use_wcs(csv)

    
    # link to the previous "fits" file solution from the input "XML" product
    msflat_obj = DpdSirRelativeFluxScaling(os.path.join(args.workdir, args.ms_flat))
    msfits = os.path.join(args.workdir, 'data' ,msflat_obj.get_sens_fits_file())
    # msflat_xml = os.path.join(args.workdir, args.ms_flat)
    # mytree = ET.parse(msflat_xml)
    # root = mytree.getroot()
    # for actor in root.findall('Data'):
    #     for a in actor.find('DataStorage'):
    #         msflat_fits = a.find('FileName').text
    # msfits = os.path.join(args.workdir,'data' ,msflat_fits)

            
    #change waveres to 2 and spatialres to 10 to just runfast
    waveres = 10 # number of wavelength points to measure
    spatialres = 68 # good to be a factor of 16
    deltaw = 350 # in Angstrom corresponds to ~44 pixels 
    ndither = 15 # for the GAIA simulation
    noisefactor = 1.0 #deep fields have less noise compared to spectra in testing, should be changed to 1 in actual code
    ncoeff = int(args.ncoeff)

    if args.calmode ==2:
        tables = LocationTableSet(args.workdir, inlocfile)    
        n = SelfCalibWave(extspecnames, list_grisms, ncoeff, args.limmag, args.calmode,args.smooth_kernel_sigma, noisefactor, deltaw, spatialres, waveres,
                      ndither, mdb, optxml, outdir=args.workdir, ms_flat = msfits, tables = tables)
    else:
        n = SelfCalibWave(extspecnames, list_grisms, ncoeff, args.limmag, args.calmode,args.smooth_kernel_sigma, noisefactor, deltaw, spatialres, waveres,
                          ndither, mdb, optxml, outdir=args.workdir, ms_flat = msfits)
    n.fit()

    #comment out next line to run fast
    #n.cross_grism() # to bring different grisms in agreement with a mean shift

    n.quality_prod() # to make a final histogram of the response for a quality assessment

    # write FITS datacontainer
    fits_filename = DpdSirRelativeFluxScaling.create_container_name('SCALE')
    fits_whole_path = os.path.join(args.workdir, 'data', fits_filename)
    logger.info(f'Write FITS file {fits_whole_path}')
    n.writeOutputSir(fits_whole_path, msfits, args.iterative)

    # write XML product
    xml_whole_path = os.path.join(args.workdir, args.output_xml)
    logger.info(f'Write XML file {fits_whole_path}')
    relflux_obj = DpdSirRelativeFluxScaling()
    relflux_obj.append_data(sens_file=fits_whole_path)
    relflux_obj.append_dqc(n.get_dq_parameters())
    relflux_obj.write(xml_whole_path)

    # ##Writing the xml and fits file
    # fits_filename = os.path.join('',FileNameProvider().get_allowed_filename(processing_function='SIR',
    #                                                                         type_name='W-RelativeFlux-SCALE',
    #                                                                        instance_id=str(1), extension='fits'))

    # fits_whole_path = os.path.join(args.workdir, "data", fits_filename)
    # n.writeOutputSir(fits_whole_path, msfits, args.iterative)


    # header = DpdRelFluxScaling.make_header()
    # product = relativeflux_stub.DpdSirRelativeFluxScaling(Header=header)
    # ValidityRange = ins_stub.calibrationValidPeriod()
    # ValidityRange.TimestampStart = '2010-01-01T00:00:00.0'
    # ValidityRange.TimestampEnd = '2050-01-01T00:00:00.0'

    # product.Data = sir_stub.relativeFluxScaling.Factory(ValidityRange=ValidityRange, CalibrationID='1', CalibrationVersion=1)
    # fits_file_storage = sir_stub.relativeFluxScalingFitsFile(format='sir.relativeFluxScaling',version='0.1')
    # fits_file_storage.DataContainer = dss_stub.dataContainer(FileName=fits_filename,filestatus="PROPOSED")
    # product.Data.DataStorage = fits_file_storage
    # product.Data.QualityParameters = n.get_dq_parameters() #JX

    # with open(os.path.join(args.workdir, args.output_xml), 'w') as ott:
    #     ott.write(product.toDOM().toprettyxml())

    logger.info('#')
    logger.info('# Exiting SIR_RelFluxModelCalibration mainMethod()')
    logger.info('#')

