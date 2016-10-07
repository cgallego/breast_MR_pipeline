# -*- coding: utf-8 -*-
"""
Created on Wed Apr 02 09:38:04 2014

Process previously selected VOI and extract dynEnh data for EMM fit
  
@ author (C) Cristina Gallego, University of Toronto
----------------------------------------------------------------------
"""

import os, os.path
import sys
import string
import datetime
from numpy import *
import dicom
from vtk.util.numpy_support import vtk_to_numpy
import vtk

from lmfit import minimize, Parameters, Parameter, report_errors, Minimizer
from lmfit.printfuncs import *
from pandas import DataFrame
import pandas as pd
import pylab

from inputs_init import *
from display import *
import processDicoms

#!/usr/bin/env python
class Dynamic(object):
    """
    Process previously selected VOI and extract dynEnh data for EMM fit.

    lmfit DynEnh EMM fitting parametersfor dynamic series data for each VOIlesions_idXXX 
    (where idXXX corresponds to table field Radiology.MassLesion.LesionFindingID)
    
    USAGE:
    =============
    loadDynamic = Dynamic()
    """
    def __init__(self): 
        """ Initialize class attributes """
        self.dynamicEMM_contour = []
        self.dynamicEMM_inside = []
        self.timepoints = [] 
        self.amp = 0
        self.alpha = 0
        self.beta = 0
        self.iAUC1 = []
        self.Slope_ini = []
        self.Tpeak = []
        self.Kpeak = []
        self.SER = []
        self.maxCr = 0 
        self.peakCr = 0
        self.UptakeRate = 0
        self.washoutRate = 0
        self.maxVr = 0 
        self.peakVr = 0
        self.Vr_increasingRate = 0 
        self.Vr_post_1 = 0
    
        
    def __call__(self):       
        """ Turn Class into a callable object """
        Dynamic()
        
    def createMaskfromMesh(self, VOI_mesh, im):
        """ Takes an image and a VOI_mesh and returns a boolean image with only 1s inside the VOI_mesh """                
        
        # Create an Image of Fext
        white_image = vtk.vtkImageData()
        white_image.DeepCopy(im) 
  
        # polygonal data --> image stencil:
        pol2stenc = vtk.vtkPolyDataToImageStencil()
        pol2stenc.SetInput(VOI_mesh)
        pol2stenc.SetOutputOrigin(im.GetOrigin())
        pol2stenc.SetOutputSpacing(im.GetSpacing())
        pol2stenc.SetOutputWholeExtent(white_image.GetExtent())
        pol2stenc.Update()
         
        # cut the corresponding white image and set the background:
        imgstenc = vtk.vtkImageStencil()
        imgstenc.SetInput(white_image)
        imgstenc.SetStencil(pol2stenc.GetOutput())
        imgstenc.ReverseStencilOff()
        imgstenc.SetBackgroundValue(0.0)
        imgstenc.Update()
        
        # write to image        
        dims = im.GetDimensions()
        scalars = imgstenc.GetOutput().GetPointData().GetScalars()
        np_scalars = vtk_to_numpy(scalars)     
        np_scalars = np_scalars.reshape(dims[2], dims[1], dims[0]) 
        np_scalars = np_scalars.transpose(2,1,0)
        
        return np_scalars, dims
        
        
    def extractfeatures_contour(self, DICOMImages, image_pos_pat, image_ori_pat, series_path, phases_series, VOI_mesh):
        """ Start pixVals for collection pixel values at VOI """
        pixVals = []
        deltaS = {}
        
        # necessary to read point coords
        VOIPnt = [0,0,0]
        ijk = [0,0,0]
        pco = [0,0,0]
        
        for i in range(len(DICOMImages)):
            abspath_PhaseID = series_path+os.sep+str(phases_series[i]) 
            print phases_series[i]
             
            # Get total number of files
            [len_listSeries_files, FileNms_slices_sorted_stack] = processDicoms.ReadDicomfiles(abspath_PhaseID)
            mostleft_slice = FileNms_slices_sorted_stack.slices[0]
            
            # Get dicom header, retrieve
            dicomInfo_series = dicom.read_file(abspath_PhaseID+os.sep+str(mostleft_slice)) 
            
            # (0008,0031) AT S Series Time            # hh.mm.ss.frac
            seriesTime = str(dicomInfo_series[0x0008,0x0031].value) 
            # (0008,0033) AT S Image Time             # hh.mm.ss.frac
            imageTime = str(dicomInfo_series[0x0008,0x0033].value)
            
            # (0008,0032) AT S Acquisition Time       # hh.mm.ss.frac
            ti = str(dicomInfo_series[0x0008,0x0032].value) 
            
            acquisitionTimepoint = datetime.time(hour=int(ti[0:2]), minute=int(ti[2:4]), second=int(ti[4:6]))
            self.timepoints.append( datetime.datetime.combine(datetime.date.today(), acquisitionTimepoint) )
            
            # find mapping to Dicom space  
            [transformed_image, transform_cube] = Display().dicomTransform(DICOMImages[i], image_pos_pat, image_ori_pat)
 
            for j in range( VOI_mesh.GetNumberOfPoints() ):
                VOI_mesh.GetPoint(j, VOIPnt)      
                
                # extract pixID at location VOIPnt
                pixId = transformed_image.FindPoint(VOIPnt[0], VOIPnt[1], VOIPnt[2])
                im_pt = [0,0,0]
                
                transformed_image.GetPoint(pixId,im_pt)           
                inorout = transformed_image.ComputeStructuredCoordinates( im_pt, ijk, pco)
                if(inorout == 0):
                    pass
                else:
                    pixValx = transformed_image.GetScalarComponentAsFloat( ijk[0], ijk[1], ijk[2], 0)
                    pixVals.append(pixValx)
                        
            # Now collect pixVals
            print "Saving %s" % 'delta'+str(i)
            deltaS['delta'+str(i)] = pixVals
            pixVals = []
                    
        print self.timepoints
        
        # Collecting timepoints in proper format
        t_delta = []
        t_delta.append(0)
        total_time = 0
        for i in range(len(DICOMImages)-1):
            current_time = self.timepoints[i+1]
            previous_time = self.timepoints[i]
            difference_time =current_time - previous_time
            timestop = divmod(difference_time.total_seconds(), 60)
            t_delta.append( t_delta[i] + timestop[0]+timestop[1]*(1./60))
            total_time = total_time+timestop[0]+timestop[1]*(1./60)
            
        # finally print t_delta
        print t_delta
        t = array(t_delta)
        print "total_time"
        print total_time
        
        ##############################################################
        # Finished sampling deltaS
        # APply lmfit to deltaS
        # first sample the mean
        data_deltaS = []; t_deltaS = []; mean_deltaS = []; sd_deltaS = []; se_deltaS = []; n_deltaS = []
        
        # append So and to
        data_deltaS.append( 0 )       
        t_deltaS.append(0)
        mean_deltaS.append( mean(deltaS['delta0']) )
        sd_deltaS.append(0)
        se_deltaS.append(0)
        n_deltaS.append( len(deltaS['delta0']) )
        
        for k in range(1,len(DICOMImages)):
            deltaS_i =  ( mean(array(deltaS['delta'+str(k)]).astype(float)) -  mean(deltaS['delta0']) )/  mean(deltaS['delta0'])
            data_deltaS.append( deltaS_i )
            t_deltaS.append(k)
            print 'delta'+str(k)
            print  data_deltaS[k]
            
            ##############################################################
            # Calculate data_error
            # estimate the population mean and SD from our samples to find SE
            # SE tells us the distribution of individual scores around the sampled mean.
            mean_deltaS_i = mean(array(deltaS['delta'+str(k)]))
            std_deltaS_i = std(array(deltaS['delta'+str(k)]))
            n_deltaS_i = len(array(deltaS['delta'+str(k)]))
                
            sd_deltaS.append( std_deltaS_i )
            mean_deltaS.append( mean_deltaS_i )
            
            # Standard Error of the mean SE
            # the smaller the variability in the data, the more confident we are that one value (the mean) accurately reflects them.
            se_deltaS.append(std_deltaS_i/sqrt(n_deltaS_i))
            n_deltaS.append(n_deltaS_i)
                        
        # make array for data_deltaS
        data = array(data_deltaS)
        
        print "\n================\nMean and SE (i.e VOI sample data)"
        print mean_deltaS
        print se_deltaS
        
        # create a set of Parameters
        params = Parameters()
        params.add('amp',   value= 10,  min=0)
        params.add('alpha', value= 1, min=0) 
        params.add('beta', value= 0.05, min=0.0001, max=0.9)
        
        # do fit, here with leastsq model
        # define objective function: returns the array to be minimized
        def fcn2min(params, t, data):
            global model, model_res, x
            """ model EMM for Bilateral DCE-MRI, subtract data"""
            # unpack parameters:
            #  extract .value attribute for each parameter
            amp = params['amp'].value    # Upper limit of deltaS
            alpha = params['alpha'].value    # rate of signal increase min-1
            beta = params['beta'].value        # rate of signal decrease min-1
                    
            model = amp * (1- exp(-alpha*t)) * exp(-beta*t)
            
            x = linspace(0, t[4], 101)
            model_res = amp * (1- exp(-alpha*x)) * exp(-beta*x)
        
            return model - data
        
        #####
        myfit = Minimizer(fcn2min,  params, fcn_args=(t,), fcn_kws={'data':data})
        myfit.prepare_fit()
        myfit.leastsq()
            
        # On a successful fit using the leastsq method, several goodness-of-fit statistics
        # and values related to the uncertainty in the fitted variables will be calculated
        print "myfit.success"
        print myfit.success
        print "myfit.residual"
        print myfit.residual
        print "myfit.chisqr"
        print myfit.chisqr
        print "myfit.redchi"
        print myfit.redchi
            
        # calculate final result
        #final = data + myfit.residual
        # write error report
        report_errors(params)
        
        # Calculate R-square
        # R_square = sum( y_fitted - y_mean)/ sum(y_data - y_mean)
        R_square = sum( (model - mean(data))**2 )/ sum( (data - mean(data))**2 )
        print "R^2"
        print R_square
        
        self.amp = params['amp'].value
        self.alpha = params['alpha'].value
        self.beta = params['beta'].value
        
        ##################################################
        # Now Calculate Extract parameters from model
        self.iAUC1 = params['amp'].value *( ((1-exp(-params['beta'].value*t[1]))/params['beta'].value) + (exp((-params['alpha'].value+params['beta'].value)*t[1])-1)/(params['alpha'].value+params['beta'].value) )
        print "iAUC1"
        print self.iAUC1
        
        self.Slope_ini = params['amp'].value*params['alpha'].value
        print "Slope_ini"
        print self.Slope_ini
    
        self.Tpeak = (1/params['alpha'].value)*log(1+(params['alpha'].value/params['beta'].value))
        print "Tpeak"
        print self.Tpeak
    
        self.Kpeak = -params['amp'].value * params['alpha'].value * params['beta'].value
        print "Kpeak"
        print self.Kpeak
    
        self.SER = exp( (t[4]-t[1])*params['beta'].value) * ( (1-exp(-params['alpha'].value*t[1]))/(1-exp(-params['alpha'].value*t[4])) )
        print "SER"
        print self.SER
        
        ##################################################
        # Now Calculate enhancement Kinetic based features
        # Based on the course of signal intensity within the lesion
        print "\n Saving %s" % 'Crk'
        So = array(deltaS['delta0']).astype(float)
        Crk = {'Cr0': mean(So)}  
        C = {}
        
        for k in range(1,len(DICOMImages)):
            Sk = array(deltaS['delta'+str(k)]).astype(float)
            print Sk
            Cr = 0
            Carray = []
            for j in range( len(So) ):
                # extract average enhancement over the lesion at each time point
                Cr = Cr + (Sk[j] - So[j])/So[j]
                Carray.append((Sk[j] - So[j])/So[j])
                
            # compile
            C['C'+str(k)] = Carray
            Crk['Cr'+str(k)] = float(Cr/len(Sk))
        
        # Extract Fii_1
        for k in range(1,5):
            currentCr = array(Crk['Cr'+str(k)]).astype(float)
            print currentCr
            if( self.maxCr < currentCr):
                self.maxCr = float(currentCr)
                self.peakCr = int(k)
                
        print "Maximum Upate (Fii_1) = %d " %  self.maxCr
        print "Peak Cr (Fii_2) = %d " %  self.peakCr
        
        # Uptake rate
        self.UptakeRate = float(self.maxCr/self.peakCr)    
        print "Uptake rate (Fii_3) "
        print self.UptakeRate
        
        # WashOut Rate
        if( self.peakCr == 4):
            self.washoutRate = 0
        else:
            self.washoutRate = float( (self.maxCr - array(Crk['Cr'+str(4)]).astype(float))/(4-self.peakCr) )
        print "WashOut rate (Fii_4) "
        print self.washoutRate


        ##################################################
        # Now Calculate enhancement-variance Kinetic based features
        # Based on Crk['Cr'+str(k)] = Cr/len(Sk)
        print "\n Saving %s" % 'Vrk'
        Vrk = {}
        
        for k in range(1,5):
            Ci = array(C['C'+str(k)]).astype(float)    
            Cri = array(Crk['Cr'+str(k)]).astype(float)
            Vr = 0
            for j in range( len(Ci) ):
                # extract average enhancement over the lesion at each time point
                Vr = Vr + (Ci[j] - Cri)**2
            # compile
            Vrk['Vr'+str(k)] = Vr/(len(Ci)-1)
        
        # Extract Fiii_1
        for k in range(1,5):
            currentVr = array(Vrk['Vr'+str(k)]).astype(float)
            if( self.maxVr < currentVr):
                print currentVr
                self.maxVr = float(currentVr)
                self.peakVr = int(k)
        
        print "Maximum Variation of enhan (Fiii_1) = %d " %  self.maxVr
        print "Peak Vr (Fii_2) = %d " %  self.peakVr
        
        # Vr_increasingRate 
        self.Vr_increasingRate = self.maxVr/self.peakVr    
        print "Vr_increasingRate (Fiii_3)" 
        print self.Vr_increasingRate
        
        # Vr_decreasingRate
        if( self.peakVr == 4):
            self.Vr_decreasingRate = 0
        else:
            self.Vr_decreasingRate = float((self.maxVr - array(Vrk['Vr'+str(4)]).astype(float))/(4-self.peakVr))
        print "Vr_decreasingRate (Fiii_4) "
        print self.Vr_decreasingRate
        
        # Vr_post_1 
        self.Vr_post_1 = float( array(Vrk['Vr'+str(1)]).astype(float))
        print "Vr_post_1 (Fiii_5)"
        print self.Vr_post_1
 
        ##################################################
        # orgamize into dataframe
        self.dynamicEMM_contour = DataFrame( data=array([[ self.amp, self.alpha, self.beta, self.iAUC1, self.Slope_ini, self.Tpeak, self.Kpeak, self.SER, self.maxCr, self.peakCr, self.UptakeRate, self.washoutRate, self.maxVr, self.peakVr, self.Vr_increasingRate, self.Vr_decreasingRate, self.Vr_post_1]]), 
                                columns=['A.contour', 'alpha.contour', 'beta.contour', 'iAUC1.contour', 'Slope_ini.contour', 'Tpeak.contour', 'Kpeak.contour', 'SER.contour', 'maxCr.contour', 'peakCr.contour', 'UptakeRate.contour', 'washoutRate.contour', 'maxVr.contour', 'peakVr.contour','Vr_increasingRate.contour', 'Vr_decreasingRate.contour', 'Vr_post_1.contour'])

        #############################################################
        # try to plot results
        pylab.figure()
        pylab.errorbar(t, data, yerr=se_deltaS, fmt='ro', label='data+SE') # data 'ro' red dots as markers
        pylab.plot(t, model, 'b', label='model')    # model fit 'b' blue
        pylab.plot(x, model_res, 'k', label='model fit')    # model fit 'k' blakc
        pylab.xlabel(" post-contrast time (min)")
        pylab.ylabel("delta S(t)")
        pylab.legend()
        
        return self.dynamicEMM_contour
        
    
    def extractfeatures_inside(self, DICOMImages, image_pos_pat, image_ori_pat, series_path, phases_series, VOI_mesh):
        """ Start pixVals for collection pixel values at VOI """
        pixVals = []
        deltaS = {}
        
        # necessary to read point coords
        VOIPnt = [0,0,0]
        ijk = [0,0,0]
        pco = [0,0,0]
        
        for i in range(len(DICOMImages)):
            abspath_PhaseID = series_path+os.sep+str(phases_series[i]) 
            print phases_series[i]
            
            # Get total number of files
            [len_listSeries_files, FileNms_slices_sorted_stack] = processDicoms.ReadDicomfiles(abspath_PhaseID)
            mostleft_slice = FileNms_slices_sorted_stack.slices[0]
            
            # Get dicom header, retrieve
            dicomInfo_series = dicom.read_file(abspath_PhaseID+os.sep+str(mostleft_slice)) 
            
            # (0008,0031) AT S Series Time            # hh.mm.ss.frac
            seriesTime = str(dicomInfo_series[0x0008,0x0031].value) 
            # (0008,0033) AT S Image Time             # hh.mm.ss.frac
            imageTime = str(dicomInfo_series[0x0008,0x0033].value)
            
            # (0008,0032) AT S Acquisition Time       # hh.mm.ss.frac
            ti = str(dicomInfo_series[0x0008,0x0032].value) 
            
            acquisitionTimepoint = datetime.time(hour=int(ti[0:2]), minute=int(ti[2:4]), second=int(ti[4:6]))
            self.timepoints.append( datetime.datetime.combine(datetime.date.today(), acquisitionTimepoint) )
            
            # find mapping to Dicom space  
            [transformed_image, transform_cube] = Display().dicomTransform(DICOMImages[i], image_pos_pat, image_ori_pat)
            
            ### Get inside of VOI            
            [VOI_scalars, VOIdims] = self.createMaskfromMesh(VOI_mesh, transformed_image)
            print "\n VOIdims"
            print VOIdims
            
            # get non zero elements
            image_scalars = transformed_image.GetPointData().GetScalars()
            numpy_VOI_imagedata = vtk_to_numpy(image_scalars)     
            
            numpy_VOI_imagedata = numpy_VOI_imagedata.reshape(VOIdims[2], VOIdims[1], VOIdims[0]) 
            numpy_VOI_imagedata = numpy_VOI_imagedata.transpose(2,1,0)
            
            print "Shape of VOI_imagedata: "
            print numpy_VOI_imagedata.shape
            
            #################### HERE GET IT AND MASK IT OUT
            self.nonzeroVOIextracted = nonzero(VOI_scalars)
            print self.nonzeroVOIextracted
            
            VOI_imagedata = numpy_VOI_imagedata[self.nonzeroVOIextracted]     
            
            print "shape of VOI_imagedata  Clipped:"
            print VOI_imagedata.shape
        
            for j in range( len(VOI_imagedata) ):
                pixValx = VOI_imagedata[j]
                pixVals.append(pixValx)
                        
            # Now collect pixVals
            print "Saving %s" % 'delta'+str(i)
            deltaS['delta'+str(i)] = pixVals
            pixVals = []
                    
        print self.timepoints
        
        # Collecting timepoints in proper format
        t_delta = []
        t_delta.append(0)
        total_time = 0
        for i in range(len(DICOMImages)-1):
            current_time = self.timepoints[i+1]
            previous_time = self.timepoints[i]
            difference_time =current_time - previous_time
            timestop = divmod(difference_time.total_seconds(), 60)
            t_delta.append( t_delta[i] + timestop[0]+timestop[1]*(1./60))
            total_time = total_time+timestop[0]+timestop[1]*(1./60)
            
        # finally print t_delta
        print t_delta
        t = array(t_delta)
        print "total_time"
        print total_time
        
        ##############################################################
        # Finished sampling deltaS
        # APply lmfit to deltaS
        # first sample the mean
        data_deltaS = []; t_deltaS = []; mean_deltaS = []; sd_deltaS = []; se_deltaS = []; n_deltaS = []
        
        # append So and to
        data_deltaS.append( 0 )       
        t_deltaS.append(0)
        mean_deltaS.append( mean(deltaS['delta0']) )
        sd_deltaS.append(0)
        se_deltaS.append(0)
        n_deltaS.append( len(deltaS['delta0']) )
        
        for k in range(1,len(DICOMImages)):
            deltaS_i =  ( mean(array(deltaS['delta'+str(k)]).astype(float)) -  mean(deltaS['delta0']) )/  mean(deltaS['delta0'])
            data_deltaS.append( deltaS_i )
            t_deltaS.append(k)
            print 'delta'+str(k)
            print  data_deltaS[k]
            
            ##############################################################
            # Calculate data_error
            # estimate the population mean and SD from our samples to find SE
            # SE tells us the distribution of individual scores around the sampled mean.
            mean_deltaS_i = mean(array(deltaS['delta'+str(k)]))
            std_deltaS_i = std(array(deltaS['delta'+str(k)]))
            n_deltaS_i = len(array(deltaS['delta'+str(k)]))
                
            sd_deltaS.append( std_deltaS_i )
            mean_deltaS.append( mean_deltaS_i )
            
            # Standard Error of the mean SE
            # the smaller the variability in the data, the more confident we are that one value (the mean) accurately reflects them.
            se_deltaS.append(std_deltaS_i/sqrt(n_deltaS_i))
            n_deltaS.append(n_deltaS_i)
                        
        # make array for data_deltaS
        data = array(data_deltaS)
        
        print "\n================\nMean and SE (i.e VOI sample data)"
        print mean_deltaS
        print se_deltaS
        
        # create a set of Parameters
        params = Parameters()
        params.add('amp',   value= 10,  min=0)
        params.add('alpha', value= 1, min=0) 
        params.add('beta', value= 0.05, min=0.0001, max=0.9)
        
        # do fit, here with leastsq model
        # define objective function: returns the array to be minimized
        def fcn2min(params, t, data):
            global model, model_res, x
            """ model EMM for Bilateral DCE-MRI, subtract data"""
            # unpack parameters:
            #  extract .value attribute for each parameter
            amp = params['amp'].value    # Upper limit of deltaS
            alpha = params['alpha'].value    # rate of signal increase min-1
            beta = params['beta'].value        # rate of signal decrease min-1
                    
            model = amp * (1- exp(-alpha*t)) * exp(-beta*t)
            
            x = linspace(0, t[4], 101)
            model_res = amp * (1- exp(-alpha*x)) * exp(-beta*x)
        
            return model - data
        
        #####
        myfit = Minimizer(fcn2min,  params, fcn_args=(t,), fcn_kws={'data':data})
        myfit.prepare_fit()
        myfit.leastsq()
            
        # On a successful fit using the leastsq method, several goodness-of-fit statistics
        # and values related to the uncertainty in the fitted variables will be calculated
        print "myfit.success"
        print myfit.success
        print "myfit.residual"
        print myfit.residual
        print "myfit.chisqr"
        print myfit.chisqr
        print "myfit.redchi"
        print myfit.redchi
            
        # calculate final result
        #final = data + myfit.residual
        # write error report
        report_errors(params)
        
        # Calculate R-square
        # R_square = sum( y_fitted - y_mean)/ sum(y_data - y_mean)
        R_square = sum( (model - mean(data))**2 )/ sum( (data - mean(data))**2 )
        print "R^2"
        print R_square
        
        self.amp = params['amp'].value
        self.alpha = params['alpha'].value
        self.beta = params['beta'].value
        
        ##################################################
        # Now Calculate Extract parameters from model
        self.iAUC1 = params['amp'].value *( ((1-exp(-params['beta'].value*t[1]))/params['beta'].value) + (exp((-params['alpha'].value+params['beta'].value)*t[1])-1)/(params['alpha'].value+params['beta'].value) )
        print "iAUC1"
        print self.iAUC1
        
        self.Slope_ini = params['amp'].value*params['alpha'].value
        print "Slope_ini"
        print self.Slope_ini
    
        self.Tpeak = (1/params['alpha'].value)*log(1+(params['alpha'].value/params['beta'].value))
        print "Tpeak"
        print self.Tpeak
    
        self.Kpeak = -params['amp'].value * params['alpha'].value * params['beta'].value
        print "Kpeak"
        print self.Kpeak
    
        self.SER = exp( (t[4]-t[1])*params['beta'].value) * ( (1-exp(-params['alpha'].value*t[1]))/(1-exp(-params['alpha'].value*t[4])) )
        print "SER"
        print self.SER
        
        ##################################################
        # Now Calculate enhancement Kinetic based features
        # Based on the course of signal intensity within the lesion
        print "\n Saving %s" % 'Crk'
        So = array(deltaS['delta0']).astype(float)
        Crk = {'Cr0': mean(So)}  
        C = {}
        Carray = []
        
        for k in range(1,len(DICOMImages)):
            Sk = array(deltaS['delta'+str(k)]).astype(float)
            Cr = 0
            for j in range( len(So) ):
                # extract average enhancement over the lesion at each time point
                Cr = Cr + (Sk[j] - So[j])/So[j]
                Carray.append((Sk[j] - So[j])/So[j])
                
            # compile
            C['C'+str(k)] = Carray
            Crk['Cr'+str(k)] = Cr/len(Sk)
        
        # Extract Fii_1
        for k in range(1,5):
            currentCr = array(Crk['Cr'+str(k)]).astype(float)
            print currentCr
            if( self.maxCr < currentCr):
                self.maxCr = float(currentCr)
                self.peakCr = int(k)
                
        print "Maximum Upate (Fii_1) = %d " %  self.maxCr
        print "Peak Cr (Fii_2) = %d " %  self.peakCr
        
        # Uptake rate
        self.UptakeRate = float(self.maxCr/self.peakCr)    
        print "Uptake rate (Fii_3) "
        print self.UptakeRate
        
        # WashOut Rate
        if( self.peakCr == 4):
            self.washoutRate = 0
        else:
            self.washoutRate = float( (self.maxCr - array(Crk['Cr'+str(4)]).astype(float))/(4-self.peakCr) )
        print "WashOut rate (Fii_4) "
        print self.washoutRate


        ##################################################
        # Now Calculate enhancement-variance Kinetic based features
        # Based on Crk['Cr'+str(k)] = Cr/len(Sk)
        print "\n Saving %s" % 'Vrk'
        Vrk = {}
        
        for k in range(1,5):
            Ci = array(C['C'+str(k)]).astype(float)    
            Cri = array(Crk['Cr'+str(k)]).astype(float)
            Vr = 0
            for j in range( len(Ci) ):
                # extract average enhancement over the lesion at each time point
                Vr = Vr + (Ci[j] - Cri)**2
            # compile
            Vrk['Vr'+str(k)] = Vr/(len(Ci)-1)
        
        # Extract Fiii_1
        for k in range(1,5):
            currentVr = array(Vrk['Vr'+str(k)]).astype(float)
            if( self.maxVr < currentVr):
                print currentVr
                self.maxVr = float(currentVr)
                self.peakVr = int(k)
        
        print "Maximum Variation of enhan (Fiii_1) = %d " %  self.maxVr
        print "Peak Vr (Fii_2) = %d " %  self.peakVr
        
        # Vr_increasingRate 
        self.Vr_increasingRate = self.maxVr/self.peakVr    
        print "Vr_increasingRate (Fiii_3)" 
        print self.Vr_increasingRate
        
        # Vr_decreasingRate
        if( self.peakVr == 4):
            self.Vr_decreasingRate = 0
        else:
            self.Vr_decreasingRate = float((self.maxVr - array(Vrk['Vr'+str(4)]).astype(float))/(4-self.peakVr))
        print "Vr_decreasingRate (Fiii_4) "
        print self.Vr_decreasingRate
        
        # Vr_post_1 
        self.Vr_post_1 = float( array(Vrk['Vr'+str(1)]).astype(float))
        print "Vr_post_1 (Fiii_5)"
        print self.Vr_post_1
 
        ##################################################
        # orgamize into dataframe
        self.dynamicEMM_inside = DataFrame( data=array([[ self.amp, self.alpha, self.beta, self.iAUC1, self.Slope_ini, self.Tpeak, self.Kpeak, self.SER, self.maxCr, self.peakCr, self.UptakeRate, self.washoutRate, self.maxVr, self.peakVr, self.Vr_increasingRate, self.Vr_decreasingRate, self.Vr_post_1]]), 
                                columns=['A.inside', 'alpha.inside', 'beta.inside', 'iAUC1.inside', 'Slope_ini.inside', 'Tpeak.inside', 'Kpeak.inside', 'SER.inside', 'maxCr.inside', 'peakCr.inside', 'UptakeRate.inside', 'washoutRate.inside', 'maxVr.inside', 'peakVr.inside','Vr_increasingRate.inside', 'Vr_decreasingRate.inside', 'Vr_post_1.inside'])

        #############################################################
        # try to plot results
        pylab.figure()
        pylab.errorbar(t, data, yerr=se_deltaS, fmt='ro', label='data+SE') # data 'ro' red dots as markers
        #pylab.plot(t, final, 'b+', label='data+residuals')    # data+residuals 'b+' blue pluses
        pylab.plot(t, model, 'b', label='model')    # model fit 'b' blue
        pylab.plot(x, model_res, 'k', label='model fit')    # model fit 'k' blakc
        pylab.xlabel(" post-contrast time (min)")
        pylab.ylabel("delta S(t)")
        pylab.legend()
        
        return self.dynamicEMM_inside