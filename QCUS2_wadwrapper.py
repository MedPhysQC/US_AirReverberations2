#!/usr/bin/env python
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# This code is an analysis module for WAD-QC 2.0: a server for automated 
# analysis of medical images for quality control.
#
# The WAD-QC Software can be found on 
# https://bitbucket.org/MedPhysNL/wadqc/wiki/Home
# 
#
# Changelog:
#   20201027: always write thumbnail
#   20200803: bugfix: do not write out ocr images if none present; separate DICOM tags for Philips/Siemens
#   20200724: bugfix: did not store dicomtags properly
#   20200722: pluginversion only once; add curved box to overview
#   20200721: Drop sensitivity analysis as it I dont know what it means; 
#             just keep max depth as it could be interesting for signal level
#   20200717: Rewrite, starting from v20200508 making a simpler module
#
# mkdir -p TestSet/StudyCurve
# mkdir -p TestSet/Config
# cp ~/Downloads/1/us_philips_*.json TestSet/Config/
# ln -s /home/nol/WAD/pyWADdemodata/US/US_AirReverberations/dicom_curve/ TestSet/StudyCurve/
# ./QCUS2_wadwrapper.py -d TestSet/StudyEpiqCurve/ -c Config/us2_philips_epiq_instance.json -r results_epiq.json
#

__version__ = '20201027'
__author__ = 'aschilham'

GUIMODE = True
GUIMODE = False
import os
# this will fail unless wad_qc is already installed
from wad_qc.module import pyWADinput
from wad_qc.modulelibs import wadwrapper_lib

import numpy as np

try:
    from scipy.misc import toimage
except (ImportError, AttributeError) as e:
    try:
        from wad_qc.modulelibs.wadwrapper_lib import toimage as toimage
    except (ImportError, AttributeError) as e:
        msg = "Function 'toimage' cannot be found. Either downgrade scipy or upgrade WAD-QC."
        raise AttributeError("{}: {}".format(msg, e))

# sanity check: we need at least scipy 0.10.1 to avoid problems mixing PIL and Pillow
import scipy
scipy_version = [int(v) for v in scipy.__version__ .split('.')]
if scipy_version[0] == 0:
    if scipy_version[1]<10 or (scipy_version[1] == 10 and scipy_version[1]<1):
        raise RuntimeError("scipy version too old. Upgrade scipy to at least 0.10.1")

if not 'MPLCONFIGDIR' in os.environ:
    import pkg_resources
    try:
        #only for matplotlib < 3 should we use the tmp work around, but it should be applied before importing matplotlib
        matplotlib_version = [int(v) for v in pkg_resources.get_distribution("matplotlib").version.split('.')]
        if matplotlib_version[0]<3:
            os.environ['MPLCONFIGDIR'] = "/tmp/.matplotlib" # if this folder already exists it must be accessible by the owner of WAD_Processor 
    except:
        os.environ['MPLCONFIGDIR'] = "/tmp/.matplotlib" # if this folder already exists it must be accessible by the owner of WAD_Processor 

import matplotlib
if not GUIMODE:
    matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.

import QCUS2_lib
import ocr_lib

def logTag():
    return "[QCUS2_wadwrapper] "

# MODULE EXPECTS PYQTGRAPH DATA: X AND Y ARE TRANSPOSED!

##### Series wrappers
def add_plugin_version(qc, results, varname):
    """
    Just add the plugin_version to the results
    """
    for res in results:
        if res.name == varname and res.category == "string":
            return

    # add results
    results.addString(varname, str(qc.qc_version))

def get_idname_from_ocr(data, params):
    """
    Separate function to generate idname from ocr, because it is needed by several functions
    """
    try:
        dummy = params['OCR_probeID:xywh']
        # build a fake action
        idparams = {}
        base = 'OCR_probeID'
        for tag in ['xywh', 'type', 'prefix', 'suffix']:
            try:
                name = '%s:%s'%(base, tag)
                val = params[name]
                idparams[name] = val
            except:
                pass
        values, error, msg = OCR(data, None, {'params':idparams}, idname='')
        if error:
            raise ValueError("Cannot find values for %s: %s"%(base, msg))
        idname = '_'+ values[base].replace('/','-')
    except Exception as e:
        idname = None # OCR cannot be found
    
    return idname

def qc_series(data, results, action, idname, override={}):
    """
    US Reverberations in Air analysis:
        Check the uniformity of the reverberation patterns

    Params needs to define:
       nothing yet
       
    Workflow:
        1. Set runtime parameters
        2. Check data format
        3. Build and populate qcstructure
        4. Run tests
        5. Build xml output
        6. Build artefact picture thumbnail
    """

    # 1.-3. 
    try:
        params = action['params']
    except KeyError:
        params = {}

    # overrides from test scripts
    for k,v in override.items():
        params[k] = v

    inputfile = data.series_filelist[0]  # give me a [filename]

    wrapper_params = [
        'rgbchannel', 'auto_suffix',
    ] # parameters for wrapper only 

    rgbchannel = params.get('rgbchannel', 'B')

    dcmInfile, pixelData, dicomMode = wadwrapper_lib.prepareInput(inputfile, headers_only=False, logTag=logTag(), rgbchannel=rgbchannel)

    # correct data order wadwrapper_lib return x,y data
    if len(pixelData.shape) == 2:
        pixelData = np.transpose(pixelData, (1,0))
    elif len(pixelData.shape) == 3:
        pixelData = np.transpose(pixelData, (0,2,1))
        
    qc = QCUS2_lib.Analysis(dcmInfile, pixelData)
    for name,value in params.items():
        if not name in wrapper_params:
            qc.set_param(name, value)
    qc.run()

    if GUIMODE:
        qc.show_report()
        print("Result in {}".format(os.getcwd()))
        matplotlib.pyplot.show()
    
    # find probename
    idname = ''
    if params.get('auto_suffix', False):
        if idname is None:
            idname = '_'+qc.imageID(probeonly=True)

    # add pluginversion to 'result' object
    add_plugin_version(qc, results, 'pluginversion'+idname)

    # add results to 'result' object
    report = qc.get_report()
    for section in report.keys():
        for key, vals in report[section].items():
            if vals[0] in ['int', 'float']: 
                results.addFloat(key+idname, vals[1])
            elif vals[0] in ['string']: 
                results.addString(key+idname, vals[1])
            elif vals[0] in ['bool']: 
                results.addBool(key+idname, vals[1])
            elif vals[0] in ['object']: 
                results.addObject(key+idname, vals[1])
            else:
                raise ValueError("Result '{}' has unknown result type '{}'".format(key, vals[0]) )


    return qc # for write_images

def acqdatetime_series(data, results, action):
    """
    Read acqdatetime from dicomheaders and write to IQC database

    Workflow:
        1. Read only headers
    """
    try:
        import pydicom as dicom
    except ImportError:
        import dicom
    try:
        params = action['params']
    except KeyError:
        params = {}

    ## 1. read only headers
    dcmInfile = dicom.read_file(data.series_filelist[0][0], stop_before_pixels=True)

    dt = wadwrapper_lib.acqdatetime_series(dcmInfile)

    results.addDateTime('AcquisitionDateTime', dt) 

def header_series(data, results, action, idname=None):
    """
    Read selected dicomfields and write to IQC database
    """
    try:
        params = action['params']
    except KeyError:
        params = {}

    inputfile = data.series_filelist[0]  # give me a [filename]

    wrapper_params = [
        'rgbchannel', 'auto_suffix',
    ] # parameters for wrapper only 

    rgbchannel = params.get('rgbchannel', 'B')
    dcmInfile, pixelData, dicomMode = wadwrapper_lib.prepareInput(inputfile, headers_only=True, logTag=logTag(), rgbchannel=rgbchannel)

    qc = QCUS2_lib.Analysis(dcmInfile, pixelData)
    for name,value in params.items():
        if not name in wrapper_params:
            qc.set_param(name, value)

    # find probename
    idname = ''
    if params.get('auto_suffix', False):
        if idname is None:
            idname = '_'+qc.imageID(probeonly=True)

    # add pluginversion to 'result' object
    add_plugin_version(qc, results, 'pluginversion'+idname)

    # run tests
    dicominfo = qc.dicom_info()
        
    # add results to 'result' object
    for dtype in dicominfo.keys():
        for key, val in dicominfo[dtype]:
            if dtype in ['int', 'float']: 
                results.addFloat(key+idname, val)
            elif dtype in ['string']: 
                results.addString(key+idname, str(val)[:min(len(str(val)),100)])
            elif dtype in ['bool']: 
                results.addBool(key+idname, val)
            elif dtype in ['object']: 
                results.addObject(key+idname, val)
            else:
                raise ValueError("Result '{}' has unknown result type '{}'".format(key, dtype) )


def OCR(qc, data, results, action, idname, override={}):
    """
    Use pyOCR which for OCR
    returns rect rois for plotting in overview
    If results is None, just return the OCR content, do not add to results
    """
    try:
        params = action['params']
    except KeyError:
        params = {}

    # overrides from test scripts
    for k,v in override.items():
        params[k] = v

    # optional parameters
    ocr_options = {}
    for lab in ['ocr_threshold', 'ocr_zoom', 'ocr_border']:
        if lab in params:
            ocr_options[lab] = int(params[lab])

    inputfile = data.series_filelist[0]  # give me a [filename]
    rgbchannel = params.get('rgbchannel', 'B')
    dcmInfile, pixelData, dicomMode = wadwrapper_lib.prepareInput(inputfile, headers_only=False, logTag=logTag(), rgbchannel=rgbchannel)
    
    # find probename
    idname = ''
    if params.get('auto_suffix', False):
        if idname is None:
            idname = '_'+qc.imageID(probeonly=True)
        
    # add pluginversion to 'result' object
    add_plugin_version(qc, results, 'pluginversion'+idname)

    rectrois = []
    error = False
    msg = ''
    values = {}
    # solve ocr params
    ocr_regions = params.get('ocr_regions',{}) # new format

    regions = {}
    for ocrname,ocrparams in ocr_regions.items():
        regions[ocrname] = {'prefix':'', 'suffix':''}
        for key,val in ocrparams.items():
            if key == 'xywh':
                regions[ocrname]['xywh'] = [int(p) for p in val.split(';')]
            elif key == 'prefix':
                regions[ocrname]['prefix'] = val
            elif key == 'suffix':
                regions[ocrname]['suffix'] = val
            elif key == 'type':
                regions[ocrname]['type'] = val

    for name, region in regions.items():
        rectrois.append([ (region['xywh'][0],region['xywh'][1]), 
                          (region['xywh'][0]+region['xywh'][2],region['xywh'][1]+region['xywh'][3])])

        txt, part = ocr_lib.OCR(pixelData, region['xywh'], **ocr_options)
        uname = name+str(idname)
        if region['type'] == 'object':
            im = toimage(part) 
            fn = '{}.jpg'.format(uname)
            im.save(fn)
            results.addObject(uname, fn)
            
        else:
            try:
                value = None
                value = ocr_lib.txt2type(txt, region['type'], region['prefix'], region['suffix'])
                if not results is None:
                    if region['type'] == 'float':
                        results.addFloat(uname, value)
                    elif region['type'] == 'string':
                        results.addString(uname, value)
                    elif region['type'] == 'bool':
                        results.addBool(uname, value)
                else:
                    values[uname] = value
            except:
                print("error", uname, value)
                error = True
                msg += uname + ' '
                im = toimage(part) 
                fn = '{}.jpg'.format(uname)
                im.save(fn)
                

    if results is None:
        return values, error, msg

    return rectrois, error, msg

def writeimages(qc, results, ocr_rois, idname):
    # also run ocr_series; needed as part of qc because of the boxes it generates
    xtra= {'rectrois': ocr_rois }
    if idname is None:
        idname = qc.label

    fname = 'overview{}.jpg'.format(idname)
    qc.save_annotated_image(fname, what='overview', xtra=xtra)
    results.addObject(os.path.splitext(fname)[0],fname)
    
def main(override={}):
    """
    override from testting scripts
    """
    data, results, config = pyWADinput()

    instances = data.getAllInstances()
    if len(instances) != 1:
        print('{} Error! Number of instances not equal to 1 ({}). Exit.'.format(logTag(),len(instances)))

    error = False
    msg = ''
    qc = None
    # read runtime parameters for module
    idname = None
    ocr_rois = []
    
    if 'ocr_series' in config['actions'].keys():
        idname = get_idname_from_ocr(data, config['actions']['ocr_series']['params'])

    for name,action in config['actions'].items():
        if name == 'acqdatetime':
            acqdatetime_series(data, results, action)

        elif name == 'header_series':
            header_series(data, results, action, idname)
        
        elif name == 'qc_series':
            qc = qc_series(data, results, action, idname, override.get('qc', {}))

        elif name == 'ocr_series':
            ocr_rois, error, msg = OCR(qc, data, results, action, idname, override.get('ocr', {}))

    #label = instance.DeviceSerialNumber+'__'+''.join(instance.TransducerData).strip()
    #if len(ocr_rois)>0:
    # always write thumbnail
    writeimages(qc, results, ocr_rois, idname)
    
    results.write()

    if error:
        raise ValueError('{} Cannot read OCR box for {}'.format(logTag(),msg))
    
    
if __name__ == "__main__":
    # main in separate function to be called by ct_tester
    main()
