# -*- coding: utf-8 -*-
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

"""
Analysis of reverberations in Air pattern.

Workflow:
input: dicomobject,pixeldata (the pixeldata is already cleaned; color is removed, leaving plain grayscale)
1. Isolate the reverberations pattern from the other information:
  a. Connected components analysis of pixelvalues >0
  b. Merge all not-to small components found along the y-axis in the middle of the imagedata, 
     excluding components fixed to the top of the image
  c. Generate a mask to isolate reverberations pattern
2. Straighten curve probe data:
  a. Detect points left and right from the middle, along the top of the reveberation pattern; if too few points
     are detected, the probe is not curved, so just crop the reveberation pattern -> rect_image
  b. Fit a circle through these points
  c. Interpolate data to straightend grid -> rect_image\
4. Uniformity analysis of rect_image to find dips that suggest problems with the probe:
  a. Make a profile along the horizontal of rect_image;
  b. Divide profile in regions: 0-10% on both sides; 10-30% on both sides; 10-90%, 30-70%; all.
  b. Count weak and dead elements (profile value below given threshold) and adjacent weak and dead elements per region.
  c. Calculate relative avg/min/max in regions
3. Sensitivity analyis of rect_image to determine vertical extend of reverberation pattern:
  a. Just return the height of the rect_image


Changelog:
    20220110: fix conversion of runtime parameters
    20200803: separate DICOM tags for Philips/Siemens
    20200724: bugfix: did not store dicomtags properly
    20200722: Add curved box to overview
    20200721: Drop sensitivity analysis as it I dont know what it means; 
              just keep max depth as it could be interesting for signal level
    20200717: Based on US_AirReverberations v20200508 (aschilham, pvanhorsen), but simpler
"""
__version__ = '20220110'
__author__ = 'aschilham'

import numpy as np
import matplotlib.pyplot as plt
import operator
import scipy

try:
    # wad2.0 runs each module stand alone
    import QCUS2_math as mymath
except ImportError:
    from . import QCUS2_math as mymath
    
from PIL import Image # image from pillow is needed
from PIL import ImageDraw # imagedraw from pillow is needed, not pil

LOCALIMPORT = False
try: 
    # try local folder
    import wadwrapper_lib
    LOCALIMPORT = True
except ImportError:
    # try wad2.0 from system package wad_qc
    from wad_qc.modulelibs import wadwrapper_lib

try:
    from scipy.misc import toimage
except (ImportError, AttributeError) as e:
    try:
        if LOCALIMPORT:
            from wadwrapper_lib import toimage as toimage
        else:
            from wad_qc.modulelibs.wadwrapper_lib import toimage as toimage
    except (ImportError, AttributeError) as e:
        msg = "Function 'toimage' cannot be found. Either downgrade scipy or upgrade WAD-QC."
        raise AttributeError("{}: {}".format(msg, e))


class QCObject:
    """
    Generic class, handles reading of DICOM tags
    """
    def __init__(self, dcmInfile, pixelData):
        self.dcmInfile = dcmInfile # dcm object for dicom tags
        self.pixelData = pixelData # pixeldata for analysis

    def readDICOMtag(self, key, imslice=0): # slice=2 is image 3
        value = wadwrapper_lib.readDICOMtag(key, self.dcmInfile, imslice)
        return value

def cropimsave(filepath, noaxis=True, fig=None):
    '''Save the current image with no whitespace
    Example filepath: "myfig.png" or r"C:\myfig.pdf" 
    '''
    if not fig:
        fig = plt.gcf()

    if noaxis:
        plt.subplots_adjust(0,0,1,1,0,0)
        for ax in fig.axes:
            ax.axis('off')
            ax.margins(0,0)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
    fig.savefig(filepath, bbox_inches='tight')
    # use  pad_inches = 0 for max cropping
    #fig.savefig(filepath, pad_inches = 0, bbox_inches='tight')

class Analysis(QCObject):
    def __init__(self, dcmInfile, pixelData):
        QCObject.__init__(self, dcmInfile, pixelData)
        self.qc_version = __version__
        self.label = "" # label to add to results
        self.origin = 'lower' # the pixeldata has the lowerleft corner labelled as 0,0
        self.origin = 'upper' # the pixeldata has the upperleft corner labelled as 0,0
        self.params = {
            # bounding box of pattern in px
            'init_pt_x0y0x1y1': None, #[158, 68, 865, 143], # in px to be used to restrict initial search of reverb pattern; three full rings (peak-to-peak); approx 3*25
            'pt_x0y0x1y1': None, #[158, 68, 865, 143], # in px to be used to restrict reverb pattern; three full rings (peak-to-peak); approx 3*25
            'f_weak': .5, #below this fraction of avg signal, the element is weak
            'f_dead': .3, #below this fraction of avg signal, the element is dead
            'signal_thresh': 0, # only pixelvalues above this number can be part of reverberations (set >0 if very noisy)
            'cluster_mode': 'all_middle', # 'largest_only' default mode of dataroi selection
            'cluster_fminsize': 10*10*3, # ignore clusters of size smaller than imwidth*imheigth/minsizefactor (wid/10*hei/10)/3
            'circle_fitfrac': 1./3, # by default use only central 1/3 of circle for fitting, as deformations towards edge can occur. 
                                      # use >1 for full fit. 1 is best for GE, 1/3 is best for Philips
            'pt_curve_radii_px': None,  #[Rc,maxrad ]
            'pt_curve_origin_px': None, #[xc,yc,Rc]
            'pt_curve_angles_deg': None, # [ang0, ang1]
            'hcor_px':0, # skip pix left and right in auto mode
            'vcor_px':0, # skip pix above and below in auto mode
        }
        self.report = {} # {'section': {'key': ( 'int', weak_num ) }}
        self.verbose = False  # dump lots of information
        self._pixmm = None    # used to convert px to mm
        self.rev_mask = None  # the reverberations mask forfull image
        self.is_curved = None # if true, then the probe is curved
        self.using_fixed_box = None # if true, then do not report found limits
        self.using_fixed_curve = None # if true, then do not report found limits
        
    def set_param(self, name, value):
        """
        set parameter name to given value, if name is a valid parameter
        """
        if name in self.params.keys():
            if name in ['f_weak', 'f_dead', 'signal_thresh', 'circle_fitfrac']:
                value = float(value)
            elif name in ['cluster_fminsize', 'hcor_px', 'vcor_px']:
                value = int(value)
            elif name in ['init_pt_x0y0x1y1', 'pt_x0y0x1y1', 'pt_curve_radii_px', 'pt_curve_origin_px', 'pt_curve_angles_deg']:
                if value == "":
                    value = None
                else:
                    value = [int(v) for v in value]
                
            self.params[name] = value
        else:
            raise ValueError("Unknown parameter '{}'".format(name))
            
    def get_report(self, section=None):
        """
        return the report dictionary. if section not is None, return only the given section.
        """
        if section is None:
            return self.report
        else:
            return self.report.get('section', [])

    def _get_sensitvity_data(self, data):
        """
        data: crop the data to a box containing the reverberation pattern
        then take only middle 33% for sensitivity analysis
        """
        hei,wid = data.shape
        
        pct_x = max(0, min(int(wid*.3+.5),wid-1))
        data = data[:, pct_x:-pct_x]
        return data
    
    def _get_uniformity_data(self, rev_mask):
        """
        crop the data to a box containing the reverberation pattern, 
        optionally removing some lines left/right and up/down
        """
        x0,y0,x1,y1 = self.params['pt_x0y0x1y1']
        dx = self.params['hcor_px']
        dy = self.params['vcor_px']
        data = (self.pixelData*rev_mask)[y0+dy:y1-dy+1,x0+dx:x1-dx+1]

        """
        Data can be enhanced by subtracting line average (and adding image average to restore abs value).
        That shows more clearly where dips and enhancements occur. Also it can be used to "isolate" the
        reverberation frequency.

        avg = np.average(data)
        davg = np.empty(data.shape, dtype=np.float)
        dprof = np.average(data, axis=1)
        for i,d in enumerate(dprof):
            davg[i,:] = data[i,:]-d+avg

        """
        
        return data

    def sensitivity(self, data):
        """
        Get a rectangular slab of the reverberation pattern.
        Average in horizontal direction.
        Just turn it into a nice image, and report the total depth of the pattern.
        """
        report_section = "sensitivity"
        self.report[report_section] = {}

        # average profile horizontally
        profile = np.average(data, axis=1)
        pos = np.array(list(range(len(profile))))
        self.report[report_section]['sens_depth_mm'] = ( 'float', self.pixels2mm(len(profile)) ) 
        if self.is_curved == True:
            if self.using_fixed_curve == False:
                self.report[report_section]['sens_depth_px'] = ( 'int', len(profile) ) 
        elif self.is_curved == False:
            if self.using_fixed_box == False:
                self.report[report_section]['sens_depth_px'] = ( 'int', len(profile) ) 
        else:
            raise ValueError("Need to set 'is_curved'!")
    
        ####
        #Analysis
        ####

        ## make the image
        # stack ring pattern on top of profile image
        xmax = np.max(pos)

        fig = plt.figure()
        ax0 = plt.axes([0.10,0.76,0.85,0.20])#, adjustable='box', aspect=myDataAspect) #x0, y0, width, height
        ax1 = plt.axes([0.10,0.10,0.85,0.65])#, adjustable='box', aspect=myDataAspect)

        # show ring pattern on top; fill whole image
        #ax0.axis('off')
        ax0.imshow(np.transpose(data,(1,0)), cmap='gray', aspect='auto')
        ax0.set_xlim(left=0, right=xmax)
        
        # make these tick labels invisible
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.setp(ax0.get_yticklabels(), visible=False)
        
        # show profile below
        ax1.plot(pos, profile, label="profile")
        ymax = np.max(profile)+1

        ax1.set_ylim(bottom=0, top=ymax)
        ax1.set_xlim(left=0, right=xmax)
        ax1.set_xlabel("position [px]")
        ax1.set_ylabel("signal [a.u.]")
        ax1.grid(which='major', axis='y')
        ax1.legend()
        
        fname = "sensitivity.jpg"
        cropimsave(fname, noaxis=False, fig=fig)
        self.report[report_section]['sensitivity'] = ( 'object', fname )
        

    def uniformity(self, data):
        """
        Get a rectangular slab of a fixed number of rings.
        Average in vertical direction.
        Count the number of pixel with response below the weak treshold, and the dead treshold.
        Separate analysis for all, the outer 10%, outer 10%-30%, and 30%-70%
        Also report the relative (to the overall mean) mean, min, max of all region 
        """
        report_section = "uniformity"
        self.report[report_section] = {}
        
        # check if 8 bit data is saturated
        pix_max = int(np.max(data)+.5)
        if pix_max>253:
            print("Warning! Saturated data. Use a lower gain!")
        
        self.report[report_section]['unif_max_pixval'] = ( 'int', pix_max )
            
        weak = self.params['f_weak']
        dead = self.params['f_dead']
        
        profile = np.average(data, axis=0)

        # report width of profile
        if self.is_curved == False:
            if self.using_fixed_box == False: 
                self.report[report_section]['unif_box_width_px'] = ( 'int', len(profile) ) 
        elif self.is_curved == True:
            if self.using_fixed_curve == False: 
                ang0,ang1 = self.params['pt_curve_angles_deg']
                self.report[report_section]['unif_curve_width_deg'] = ( 'float', ang1-ang0 ) 
        else:
            raise ValueError("Need to set 'is_curved'!")
        self.report[report_section]['unif_width_mm'] = ( 'float', self.pixels2mm(len(profile)) ) 

        mean = np.average(profile)
        weak = weak*mean
        dead = dead*mean
        
        # analyse
        weak_idx = sorted(np.where(profile<weak)[0])
        dead_idx = sorted(np.where(profile<dead)[0])

        def _analyse_uniformity_idx(idxs):
            """
            count number of elements in list and number of neighboring elements
            """
            neighbors = 0
            if len(idxs)>0:
                idxs = sorted(idxs)
                for i in range(len(idxs)-1):
                    if idxs[i+1] == idxs[i]+1:
                        neighbors += 1

            return len(idxs), neighbors
        
        # analyse all and in buckets
        num = len(profile)
        idx = list(range(num))
        pct_10 = max(0, min(int(num*.1+.5),num-1))
        pct_30 = max(0, min(int(num*.3+.5),num-1))
        buckets = [
            ('all', idx),
            ('*00_10', idx[:pct_10]+idx[-pct_10:]),
            ('*10_30', idx[pct_10:pct_30]+idx[-pct_30:-pct_10]),
            ('10_90', idx[pct_10:-pct_10]),
            ('30_70', idx[pct_30:-pct_30]),
        ]

        for lab, idx_valid in buckets:
            weak_num, weak_neighbors = _analyse_uniformity_idx([i for i in weak_idx if i in idx_valid])
            dead_num, dead_neighbors = _analyse_uniformity_idx([i for i in dead_idx if i in idx_valid])
            self.report[report_section]['unif_weak_{}'.format(lab)] = ( 'int', weak_num ) 
            self.report[report_section]['unif_weaknbs_{}'.format(lab)] = ( 'int', weak_neighbors ) 
            self.report[report_section]['unif_dead_{}'.format(lab)] = ( 'int', dead_num ) 
            self.report[report_section]['unif_deadnbs_{}'.format(lab)] = ( 'int', dead_neighbors ) 
            loc_prof = profile[idx_valid]
            if not lab == 'all':
                self.report[report_section]['unif_relmean_{}'.format(lab)] = ( 'float', np.average(loc_prof)/mean)
                self.report[report_section]['unif_relmin_{}'.format(lab)] = ( 'float', np.min(loc_prof)/mean )
                self.report[report_section]['unif_relmax_{}'.format(lab)] = ( 'float', np.max(loc_prof)/mean )
        
        # fill report with other results
        self.report[report_section]['unif_mean'] = ( 'float', mean ) 
        self.report[report_section]['unif_relmin'] = ( 'float', np.min(profile)/mean )
        self.report[report_section]['unif_relmax'] = ( 'float', np.max(profile)/mean )
        self.report[report_section]['unif_f_weak'] = ( 'float', self.params['f_weak'] )
        self.report[report_section]['unif_f_dead'] = ( 'float', self.params['f_dead'] )
        
        # stack ring pattern on top of profile image
        pos = np.array(list(range(len(profile))))
        fig = plt.figure()
        #plt.title("uniformity")
        ax0 = plt.axes([0.10,0.76,0.85,0.20])#, adjustable='box', aspect=myDataAspect) #x0, y0, width, height
        ax1 = plt.axes([0.10,0.10,0.85,0.65])#, adjustable='box', aspect=myDataAspect)

        # show ring pattern on top; fill whole image
        #ax0.axis('off')
        ax0.imshow(data, cmap='gray', aspect='auto')
        
        # make these tick labels invisible
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.setp(ax0.get_yticklabels(), visible=False)
        
        # show profile below
        ax1.plot(pos, profile, label="profile")
        ax1.plot([pos[0], pos[-1]], [mean, mean], linestyle=':', linewidth=2, color='green', label="average")
        ax1.plot([pos[0], pos[-1]], [weak, weak], linestyle=':', linewidth=2, color='orange', label="weak")
        ax1.plot([pos[0], pos[-1]], [dead, dead], linestyle=':', linewidth=2, color='red', label="dead")

        # add buckets
        ymax = np.max(profile)+1
        xmax = np.max(pos)
        ax1.axvspan(pos[ 0], pos[ pct_10], facecolor='black', alpha=0.2)
        ax1.axvspan(pos[-1], pos[-pct_10], facecolor='black', alpha=0.2)
        ax1.axvspan(pos[ pct_10], pos[ pct_30], facecolor='black', alpha=0.1)
        ax1.axvspan(pos[-pct_30], pos[-pct_10], facecolor='black', alpha=0.1)
        ax1.set_ylim(bottom=0, top=ymax)
        #ax1.set_xlim(left=0, right=xmax)
        offset=5
        ax1.set_xlim(left=-offset, right=xmax+offset) # want to see first and last line
        ax0.set_xlim(left=-offset, right=xmax+offset) # want to see first and last line
        ax1.set_xlabel("position [px]")
        ax1.set_ylabel("signal [a.u.]")
        ax1.grid(which='major', axis='y')
        ax1.legend()
            
        fname = "uniformity.jpg"
        cropimsave(fname, noaxis=False, fig=fig)
        self.report[report_section]['uniformity'] = ( 'object', fname )

        
        
    def show_report(self):
        """
        just print results in a table format
        """
        for hdr,vals in self.report.items():
            print("Section: {}".format(hdr))

            for key,val in vals.items():
                print("\t{}\t{}\t{}".format(key, val[0], val[1]))

    def dicom_info(self, info='dicom'):
        # Different from ImageJ version; tags "0008","0104" and "0054","0220"
        #  appear to be part of sequences. This gives problems (cannot be found
        #  or returning whole sequence blocks)
        # Possibly this can be solved by using if(type(value) == type(dicom.sequence.Sequence()))
        #  but I don't see the relevance of these tags anymore, so set them to NO

        import string
        printable = set(string.printable)
        
        try:
            manufacturer = str(self.readDICOMtag("0008,0070")).lower()
            if "siemens" in manufacturer:
                manufacturer = "Siemens"
            elif "philips" in manufacturer:
                manufacturer = "Philips"
            else:    
                print("Manufacturer '{}' not implemented. Treated as 'Philips'.".format(manufacturer))
                manufacturer = "Philips"
        except:
            print("Could not determine manufacturer. Assuming 'Philips'.")
            
        if manufacturer == "Philips":
            if info == "dicom":
                dicomfields = {
                    'string': [
                        ["0008,0012", "Instance Date"],
                        ["0008,0013", "Instance Time"],
                        ["0008,0060", "Modality"],
                        ["0008,0070", "Manufacturer"],
                        ["0008,1090", "Manufacturer Model Name"],
                        ["0008,1010", "Station Name"],
                        ["0008,1030", "Study Description"],
                        ["0008,0068", "Presentation Intent Type"], 
                        ["0018,1000", "Device Serial Number"],
                        ["0018,1020", "Software Version(s)"],
                        ["0018,1030", "Protocol Name"],
                        ["0018,5010", "Transducer Data"],
                        ["0018,5020", "Processing Function"],
                        ["0028,2110", "Lossy Image Compression"],
                        ["2050,0020", "Presentation LUT Shape"],
                        ],
                    'float': [
                        ["0028,0002", "Samples per Pixel"],
                        ["0028,0101", "Bits Stored"],
                        ["0018,6011, 0018,6024", "Physical Units X Direction"],
                        ["0018,6011, 0018,602c", "Physical Delta X"],
                        ] # Philips
                    }
            elif info == "id":
                dicomfields = {
                    'string': [
                        ["0008,1010", "Station Name"],
                        ["0018,5010", "Transducer"],
                        ["0008,0012", "InstanceDate"],
                        ["0008,0013", "InstanceTime"]
                    ]
                }
 
            elif info == "probe":
                dicomfields = {
                    'string': [
                        ["0018,5010", "Transducer"],
                    ]   
                }

        elif manufacturer == "Siemens":
            if info == "dicom":
                dicomfields = {
                    'string': [
                        ["0008,0023", "Image Date"],
                        ["0008,0033", "Image Time"],
                        ["0008,0060", "Modality"],
                        ["0008,0070", "Manufacturer"],
                        ["0008,1090", "Manufacturer Model Name"],
                        ["0008,1010", "Station Name"],
                        ["0018,1000", "Device Serial Number"],
                        ["0018,1020", "Software Version(s)"],
                        ["0018,5010", "Transducer Data"],
                    ],
                    'float': [
                        ["0028,0002", "Samples per Pixel"],
                        ["0028,0101", "Bits Stored"],
                        ["0018,6011, 0018,6024", "Physical Units X Direction"],
                        ["0018,6011, 0018,602c", "Physical Delta X"],
                        ["0018,5022", "Mechanical Index"],
                        ["0018,5024", "Thermal Index"],
                        ["0018,5026", "Cranial Thermal Index"],
                        ["0018,5027", "Soft Tissue Thermal Index"],
                        ["0019,1003", "FrameRate"],
                        ["0019,1021", "DynamicRange"],
                    ] # Siemens
                }
            

            elif info == "id":
                dicomfields = {
                    'string': [
                        ["0008,1010", "Station Name"],
                        ["0018,5010", "Transducer"],
                        ["0008,0023", "ImageDate"],
                        ["0008,0033", "ImageTime"],
                    ]
                }
 
            elif info == "probe":
                dicomfields = {
                    'string': [
                        ["0018,5010", "Transducer"],
                    ]   
                }

        results = {}
        for dtype in dicomfields.keys():
            if not dtype in results.keys():
                results[dtype] = []
            for df in dicomfields[dtype]:
                key = df[0]
                value = ""
                try:
                    value = str(self.readDICOMtag(key)).replace('&','')
                    value = ''.join(list(filter(lambda x: x in printable, value)))
                except:
                    value = ""
                    
                if dtype in ['string']:
                    results[dtype].append( (df[1], value) )
                elif dtype in ['int']:
                    results[dtype].append( (df[1], int(value))  )
                elif dtype in ['float']:
                    results[dtype].append( (df[1], float(value))  )

        return results


    def imageID(self, probeonly=False):
        """
        find a identifyable suffix
        """
        if not self.label in [None, ""]:
            return self.label

        # make an identifier for this image
        if probeonly:
            di = self.dicom_info(info='probe')
        else:
            di = self.dicom_info(info='id')

        # construct label from tag values
        label = '_'.join(v for k,v in di['string'])

        # sanitize label
        forbidden = '[,]\'" '
        label2 = ''
        for la in label:
            if la in forbidden:
                continue
            else:
                label2 += la
        label2 = label2.replace('UNUSED', '') # cleaning
        label2.replace('/','-')

        self.label = label2
        return label2

    def save_annotated_image(self, fname, what='overview', xtra={}):
        """
        Make an jpg of the original image, indicating all analyzed and interesting parts
        """
        # make a palette, mapping intensities to greyscale
        pal = np.arange(0,256,1,dtype=np.uint8)[:,np.newaxis] * \
            np.ones((3,),dtype=np.uint8)[np.newaxis,:]
        # but reserve the first for red for markings
        pal[0] = [255,0,0]

        rectrois = []
        polyrois = []
        circlerois = []

        # convert to 8-bit palette mapped image with lowest palette value used = 1
        if what == 'overview':
            # first the base image
            work = np.array(self.pixelData)
            work[self.pixelData ==0] = 1
            im = toimage(work, low=1, pal=pal)

            # add box around reverb region
            x0,y0,x1,y1 = self.params['pt_x0y0x1y1']
            if self.is_curved == False:
                rectrois.append( [(x0, y0),(x1, y1)] )
            
            # add curved pattern box
            if self.is_curved == True:
                curve_roi = []
                ang0,ang1 = self.params['pt_curve_angles_deg']
                r0,r1     = self.params['pt_curve_radii_px']
                xc,yc,rc = self.params['pt_curve_origin_px'] #[xc,yc,Rc]

                for ang in np.linspace(ang0, ang1, num=x1-x0, endpoint=True):
                    x = xc+r0*np.sin(np.pi/180.*ang)
                    y = yc+r0*np.cos(np.pi/180.*ang)
                    curve_roi.append((x,y))
                for ang in np.linspace(ang1, ang0, num=x1-x0, endpoint=True):
                    x = xc+r1*np.sin(np.pi/180.*ang)
                    y = yc+r1*np.cos(np.pi/180.*ang)
                    curve_roi.append((x,y))
                polyrois.append(curve_roi)

        # add extra rois if provided
        if 'circlerois' in xtra:
            for r in xtra['circlerois']:
                circlerois.append(r)
        if 'polyrois' in xtra:
            for r in xtra['polyrois']:
                polyrois.append(r)
        if 'rectrois' in xtra:
            for r in xtra['rectrois']:
                rectrois.append(r)

        # now draw all rois in reserved color
        draw = ImageDraw.Draw(im)
        for r in polyrois:
            #[ [ (x,y) ] ]
            roi =[]
            for x,y in r:
                roi.append( (int(x+.5),int(y+.5)))
            draw.polygon(roi,outline=0)

        for r in rectrois:
            # [ (x0,y0),(x1,y1) ]
            (x0,y0),(x1,y1) = r
            draw.rectangle(((x0,y0),(x1,y1)),outline=0)

        # now draw all cirlerois in reserved color
        for x,y,r in circlerois:
            # [ (x,y,r) ]
            draw.ellipse((x-r,y-r,x+r,y+r), outline=0)
        del draw

        # convert to RGB for JPG, cause JPG doesn't do PALETTE and PNG is much larger
        im = im.convert("RGB")

        imsi = im.size
        if max(imsi)>2048:
            ratio = 2048./max(imsi)
            im = im.resize( (int(imsi[0]*ratio+.5), int(imsi[1]*ratio+.5)),Image.ANTIALIAS)
        im.save(fname)

    def pixels2mm(self, px):
        """
        translate pixels into mm
        """
        if not self._pixmm is None:
            return self._pixmm*px

        dicomfields = [
            ["0018,6011, 0018,6024", "Physical Units X Direction"],
            ["0018,6011, 0018,602c", "Physical Delta X"],
            ] # Philips

        # make sure the provided dicom units are cm
        units = self.readDICOMtag(dicomfields[0][0])
        if units != 3: # 3 = cm
            return -1

        mm = 10.*self.readDICOMtag(dicomfields[1][0]) # convert to mm
        self._pixmm = mm

        return px*mm

    def isolate_reverberations(self):
        """
        Find reverbrations part of image.
        Workflow:
        1. Restrict to bbox if provided
        2. Find reverberations as largest connected component != 0
        2. Return reverb mask
        """
        report_section = "pattern"
        if not report_section in self.report.keys():
            self.report[report_section] = {}
            
        # cluster connected components with pixelvalues>0
        #work = (cs.pixeldataIn>0) * (cs.pixeldataIn<255) # must give a signal, but 255 often reserved for overlay
        work = self.pixelData>self.params['signal_thresh']

        # restrict to bbox if provided:
        self.using_fixed_box = False
        if not self.params['pt_x0y0x1y1'] is None:
            xmin,ymin,xmax,ymax = self.params['pt_x0y0x1y1'] # provided bbox
            work[      :ymin,       :    ] = 0
            work[ymax+1:    ,       :    ] = 0
            work[      :    ,       :xmin] = 0
            work[      :    , xmax+1:    ] = 0
            self.using_fixed_box = True
        elif not self.params['init_pt_x0y0x1y1'] is None:
            xmin,ymin,xmax,ymax = self.params['init_pt_x0y0x1y1'] # provided initial bbox
            work[      :ymin,       :    ] = 0
            work[ymax+1:    ,       :    ] = 0
            work[      :    ,       :xmin] = 0
            work[      :    , xmax+1:    ] = 0
            
        cca = wadwrapper_lib.connectedComponents()
        cca_image, nb_labels = cca.run(work)

        if self.params['cluster_mode'] == 'largest_only': # model of PVH
            # select only largest cluster
            cluster_sizes = cca.clusterSizes()
            clus_val = np.argmax(cluster_sizes)
            rev_mask = (cca_image == clus_val)
            clus = cca.indicesOfCluster(clus_val)
        else: #'all_middle'
            # select all clusters present in vertical middle area of image, excluding top and 0
            hei,wid = np.shape(cca_image)
            # first remove very small clusters (can be located around top edge!)
            minsize = wid*hei*self.params['cluster_fminsize'] #(wid/10*hei/10)/3
            while sum(cca.clusterSizes()> minsize)<2 and minsize>100:
                minsize = int(minsize/10)
            #cca.removeSmallClusters(wid/10*hei/10)
            cca.removeSmallClusters(minsize)
            search = cca_image[:,int(0.4*wid):int(0.6*wid)]
            labs = []
            for ss in search.ravel():
                if ss>0 and not ss in labs:
                    labs.append(ss)
            # exclude labels in top rows (never part of imagedata, but full of annotations)
            search = cca_image[:,0:5]
            notlabs = []
            for ss in search.ravel():
                if ss>0 and not ss in notlabs:
                    notlabs.append(ss)
            labs = [la for la in labs if la not in notlabs]

            rev_mask = np.reshape(np.in1d(cca_image,labs),np.shape(cca_image))
            clus = np.where(rev_mask)
            clus = [ (x,y) for x,y in zip(clus[0],clus[1]) ]


        if self.verbose:
            # make an image of only largest cluster applied to original pixeldata
            reverb_image = self.pixelData*rev_mask
            toimage(reverb_image).save('reverbimage.jpg')

        # bounds of reverb image
        if len(clus) == 0:
            return None
        else:
            rev_minx = min(clus,key=operator.itemgetter(1))[1]
            rev_maxx = max(clus,key=operator.itemgetter(1))[1]
            rev_maxy = max(clus,key=operator.itemgetter(0))[0]
            rev_miny = min(clus,key=operator.itemgetter(0))[0]
            if self.params.get('pt_x0y0x1y1', None) is None:
                self.params['pt_x0y0x1y1'] = [
                    rev_minx, rev_miny, rev_maxx, rev_maxy
                ]
                self.report[report_section]['box_xmin_px'] = ('int', rev_minx)
                self.report[report_section]['box_ymin_px'] = ('int', rev_miny)
                self.report[report_section]['box_xmax_px'] = ('int', rev_maxx)
                self.report[report_section]['box_ymax_px'] = ('int', rev_maxy)

        return rev_mask

    def find_curve(self, rev_mask):
        """
        Straighten curved reverb image if applicable
        Workflow:
        1. Fit circle through top of reverb data
        """
        report_section = "curved"
        self.report[report_section] = {}
        """
        ('Curve_X',cs.curve_xyr[0], 2),
        ('Curve_Y',cs.curve_xyr[1], 2),
        ('Curve_R',cs.curve_xyr[2], 2),
        ('Curve_Residu',cs.curve_residu, 2),
        ('Curve_OpenDeg',cs.curve_opendeg, 2),
        ('Rect_width',np.shape(cs.rect_image)[0], 2),
        """
        was_curved = True
        curve_radii  = self.params.get('pt_curve_radii_px', None)  #[Rc,maxrad ]
        curve_xyr    = self.params.get('pt_curve_origin_px', None) #[xc,yc,Rc]
        curve_angles = self.params.get('pt_curve_angles_deg', None) # [ang0, ang1]
        self.using_fixed_curve = True
        if None in [curve_radii, curve_xyr, curve_angles]:
            self.using_fixed_curve = False
            ## 2. Transform reverb data to rectangle if needed
            ## Fit a circle to top of reverb pattern
            # From top of reverb image down, look for pixels != 0 from mid to left and from mid to right;
            # if both found, add it to the list
            circL_xy = []
            circR_xy = []
            x0,y0,x1,y1 = self.params['pt_x0y0x1y1']
            
            midx = int(0.5*(x0+x1)+.5)
            for y in range(y0,y1+1):
                for x in reversed(range(x0, midx)): #find left point
                    xl = -1
                    xr = -1
                    if rev_mask[y,x]:
                        xl = x
                        break
                for x in range(midx, x1): #find right point
                    if rev_mask[y,x]:
                        xr = x
                        break
                if xl>-1 and xr>-1:
                    circL_xy.append((xl,y))
                    circR_xy.append((xr,y))
                    if xr-xl<10: # stop looking
                        break
            circ_xy = []
            circ_xy.extend(circL_xy)
            circ_xy.extend(circR_xy)
            circ_xy.sort(key=operator.itemgetter(1))
            circle_fitfrac = self.params['circle_fitfrac']
            if len(circ_xy)<11:
                # at least 10 point, else probably not a curved probe
                was_curved = False
                return was_curved

            # use only central part for fitting, as deformations towards edges occur
            if circle_fitfrac<1 and circle_fitfrac>0:
                fff = 1.-circle_fitfrac
                cf = mymath.CircleFit(circ_xy[int(fff*len(circ_xy)):])
            else:
                cf = mymath.CircleFit(circ_xy) # seems best for GE

            fit = cf.fit()
            (xc,yc) = fit[0]
            Rc = fit[1]

            # calculate limiting angles and radii
            curve_angles = [np.arctan2(circL_xy[0][0]-xc,circL_xy[0][1]-yc),np.arctan2(circR_xy[0][0]-xc,circR_xy[0][1]-yc)]
            maxrad = min( [
                (x0-xc)/np.sin(curve_angles[0]),
                (x1-xc)/np.sin(curve_angles[1]),
                (y1-yc)
            ])
            self.params['pt_curve_radii_px']  = [ Rc,maxrad ]
            self.params['pt_curve_origin_px'] = [ xc,yc,Rc ]
            self.params['pt_curve_angles_deg'] = [ c/np.pi*180. for c in curve_angles]

            self.report[report_section]['curve_residu'] = ('float',  cf.residu)
            self.report[report_section]['curve_radmin_px'] = ('float',  self.params['pt_curve_radii_px'][0])
            self.report[report_section]['curve_radmax_px'] = ('float',  self.params['pt_curve_radii_px'][1])
            self.report[report_section]['curve_xc_px'] = ('float',  xc)
            self.report[report_section]['curve_yc_px'] = ('float',  yc)
            self.report[report_section]['curve_Rc_px'] = ('float',  Rc)
            self.report[report_section]['curve_angmin_deg'] = ('float',  self.params['pt_curve_angles_deg'][0])
            self.report[report_section]['curve_angmax_deg'] = ('float',  self.params['pt_curve_angles_deg'][1])

        return was_curved

    def straighten_curve(self, rev_mask):
        """
        transform reverb pattern to rectangle: interpolate at coords
        """
        image = self.pixelData*rev_mask

        x0,y0,x1,y1 = self.params['pt_x0y0x1y1']
        curve_angles = [ c/180.*np.pi for c in self.params['pt_curve_angles_deg'] ]
        curve_radii  = self.params['pt_curve_radii_px']  #[Rc,maxrad ]
        curve_xyr    = self.params['pt_curve_origin_px'] #[xc,yc,Rc]

        ang = np.linspace(curve_angles[0], curve_angles[1], x1-x0)
        rad = np.linspace(curve_radii[0], curve_radii[1], int(0.5+curve_radii[1]-curve_radii[0]))
        an,ra = scipy.meshgrid(ang,rad)

        xi = curve_xyr[0]+ra*np.sin(an)
        yi = curve_xyr[1]+ra*np.cos(an)

        coords = np.array( [yi,xi] )
        rect_image = scipy.ndimage.map_coordinates(image, coords)
        if self.verbose:
            toimage(rect_image).save('uncurvedreverb.jpg')

        return rect_image


    def run(self):
        """
        wrapper to start all analysis steps
        """
        self.is_curved = False
        
        # isolate the reverb pattern
        rev_mask = self.isolate_reverberations()
        if rev_mask is None:
            raise ValueError("Error! Could not isolate reverberations.")

        # 2. Transform reverb data to rectangle if needed
        self.is_curved = self.find_curve(rev_mask)
        if self.is_curved == True: # could be None!
            unidata = self.straighten_curve(rev_mask)
        else:
            # get a rectangular slab of data containing only data to analyse
            unidata = self._get_uniformity_data(rev_mask)
        
        # 3. uniformity analysis: per crystal find response
        self.uniformity(unidata)
        
        # 4. sensitivity analysis: find decay and wavelength
        sensdata = self._get_sensitvity_data(unidata)
        self.sensitivity(sensdata)

        
