#
# Copyright (C) 2012-2020 Euclid Science Ground Segment
#
# This library is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser General
# Public License as published by the Free Software Foundation; either version 3.0 of the License, or (at your option)
# any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along with this library; if not, write to
# the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
#

#
# File RelativeFluxCorrectionMain.py
#
# Created on: Jul 14, 2020
#
# Author: Marco Fumana, Shooby
#
# How to run:
#
# ERun SIR_SelfCalib RelativeFluxCorrectionMain \
#      --workdir=/home/shemmati/Work/Projects/Applyflat \
#      --col_spectra=extspecs.json \
#      --ms_flat=fake_msflat.fits
#
#

import os
import json
import argparse

from astropy.io import fits

import ElementsKernel.Logging as log

#from SIR_Utilities.Formatters import str_to_bool

from SIR_H5SpectraExtractionBinding import H5ExtractedSpectrum
#from SIR_SpectraExtraction.DpdExtractedSpectraCollection import DpdExtractedSpectraCollection
from SIR_SpectraExtraction.H5ExtractedSpectraCollection import H5ExtractedSpectraCollection
from SIR_Products.DpdSirRelativeFluxScaling import DpdSirRelativeFluxScaling
from SIR_Products.DpdSirExtractedSpectraCollection import DpdSirExtractedSpectraCollection


from .RelFluxDq import RelFluxDq #JX
from .RelativeFluxCorrectionCore import RelativeFluxCorrectionCore


def defineSpecificProgramOptions():
    """
    Defines the (command line and configuration file) options specific to this task

    Returns:
      An  ArgumentParser.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--ver', action='version', version='0.1')
    parser.add_argument('--workdir', type=str, required=True,
                        help='The root working directory where the data is located')
    parser.add_argument("--logdir", type=str, required=False, default='/dev/null',
                         help="The logging directory")
    parser.add_argument('--extr_spectra_list', type=str, required=True,
                        help='The spectrum collection json list (for IAL purposes) or single XML file')
    parser.add_argument('--output_list',type=str,required=True,
                        help='Output spectra collection json list')
    parser.add_argument('--ms_flat', type=str, required=True,
                        help='The Master Flat cube')
    parser.add_argument('--mdbfile', type=str, required=True,
                        help='The MDB file')
    parser.add_argument('--config',type=str, required=True,
                        help='Config file')
    parser.add_argument('--overwrite', action='store_true', #nargs='?', dest='overwrite', default=False, const=True, type=str_to_bool,
                        help='Overwrite the corrected dataset if already exists')
    return parser


def mainMethod(args):

    module = 'RelativeFluxCorrectionMain'
    logger = log.getLogger(module)
    logger.info(f'Entering {module} mainMethod()')

    grisms_names = ['RGS000','RGS180+4','RGS000-4','RGS180','RGS270','BGS000']
    list_grisms_main =[0,6,-4,2,1,3] ### double check when blue grism and RGS270 data exists

    in_xml = []
    with open(os.path.join(args.workdir, args.extr_spectra_list), 'r') as j_file:
        try:
            in_xml = json.load(j_file)
            logger.info(f'Number of products to be calibrated {len(in_xml)}')

        except json.JSONDecodeError :
            in_xml = [args.extr_spectra_list]
            logger.info(f'Calibrating {args.extr_spectra_list}')

    # loop all XML products
    out_xml_files = []
    for xm in in_xml:
        # get HDF5 filename from XML
        extract_obj = DpdSirExtractedSpectraCollection(os.path.join(args.workdir, xm))
        h5file_name = os.path.join(args.workdir, 'data', extract_obj.get_hdf5_filename())
        logger.info(f'Loading {h5file_name}')
        h5file = H5ExtractedSpectraCollection.load(h5file_name, ids=[H5ExtractedSpectrum.SPEC1D_LABEL,])

        if (h5file.getSize() > 0):
            grism_param = h5file.getGWAPosition()+h5file.getGWATilt()
            grisname = grisms_names[list_grisms_main.index(grism_param)]

            rel_flux_obj = DpdSirRelativeFluxScaling(os.path.join(args.workdir, args.ms_flat))
            ms_flat_name = os.path.join(args.workdir, 'data', rel_flux_obj.get_sens_fits_file())

            with fits.open(ms_flat_name) as ms_flat:

                ff_correction = RelativeFluxCorrectionCore(ms_flat[grisname+'.SCI'].data,ms_flat[grisname+'.DQ'].data)

                spectra = h5file.getExtractedSpectra()
                wavetest = spectra[0].getWavelength()
                wstart,wend = wavetest[0],wavetest[-1]
                ff_correction.setup_interpolators(wstart, wend)

                flag = ff_correction.invoke(h5file,args.overwrite)

            H5ExtractedSpectraCollection.appendSpectrumDataSet(h5file, h5file_name,
                                                               ids=RelativeFluxCorrectionCore.SPEC1D_RELFLUX_LABEL,
                                                               replace=True)
            logger.info(f"{h5file_name} updated")


            #JX Compute Data Qualities
            dq = RelFluxDq()
            dq.set_flag(flag)

            # Create a NEW XML product for REL corrected spectra collections
            xml_outfile_new_name =  args.output_list.replace('.json', '.xml')
            xml_outfile_new_name_path =  os.path.join(args.workdir,xml_outfile_new_name)
            logger.info(f'Writing XML product {xml_outfile_new_name_path}')
            abs_corr_obj = DpdSirExtractedSpectraCollection()
            abs_corr_obj.append_data(h5file=h5file_name)
            # move previous ExtractedSpectraCollection product DQC(s) in his new one
            abs_corr_obj.append_dqc(dqc_parameters=extract_obj.get_dqc())
            # append REL DQC
            abs_corr_obj.append_dqc(dqc_parameters=dq.get_dict_dqc_parameters())
            abs_corr_obj.write(xml_outfile_new_name_path)

            out_xml_files.append(xml_outfile_new_name)

    # write JSON file
    out_list_name = os.path.join(args.workdir, args.output_list)
    logger.info(f'Writing {out_list_name}')
    with open(out_list_name,'w') as out_file:
        json.dump(out_xml_files, out_file)

    # the end
    logger.info(f'{module} Done.')
