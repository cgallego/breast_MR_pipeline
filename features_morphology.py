# -*- coding: utf-8 -*-
"""
Created on Wed Apr 02 09:39:48 2014

@ author (C) Cristina Gallego, University of Toronto, 2013
----------------------------------------------------------------------
"""
import os
import os.path
import sys
import string
from sys import argv, stderr, exit

from numpy import *
import vtk
from pandas import DataFrame

from operator import itemgetter, attrgetter
from vtk.util.numpy_support import vtk_to_numpy

from scipy import stats
from scipy.ndimage import *
from pylab import *
import matplotlib.pyplot as plt
from display import *

#!/usr/bin/env python
class Morphology(object):
    """
    Process previously selected VOI and extract dynEnh data for EMM fit.

    lmfit DynEnh EMM fitting of parameters for dynamic series data for each VOIlesions_idXXX 
    (where idXXX corresponds to table field Radiology.MassLesion.LesionFindingID)
    
    USAGE:
    =============
    loadMorphology = Morphology()
    """
    def __init__(self): 
        """ Initialize class attributes """
        self.morphologyFeatures = []
        self.allmin_F_r_i = []
        self.allmax_F_r_i = []
        self.allmean_F_r_i = []
        self.allvar_F_r_i = []
        self.allskew_F_r_i = []
        self.allkurt_F_r_i = []
        self.i_var_max = 0
        self.ii_var_min = 0
        self.iii_var_max = 0
        self.iii_var_max_k = 0
        self.ivVariance = 0
        self.circularity = 0
        self.irregularity = 0
        self.edge_sharp_mean = 0
        self.edge_sharp_std = 0
        self.max_RGH_mean = 0
        self.max_RGH_mean_k = 0
        self.max_RGH_var = 0
        self.max_RGH_var_k = 0
        
        
    def __call__(self):       
        """ Turn Class into a callable object """
        Morphology()
        
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
        pol2stenc.SetInformationInput(white_image)
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
        
        return np_scalars
        
        
    def extractfeatures(self, DICOMImages, image_pos_pat, image_ori_pat, series_path, phases_series, VOI_mesh):
        """ Start pixVals for collection pixel values at VOI """
        pixVals_margin = []; pixVals = []
        Fmargin = {}; voxel_frameS = {}
        
        # necessary to read point coords
        VOIPnt = [0,0,0]
        ijk = [0,0,0]
        pco = [0,0,0]
        
     
        for i in range(len(DICOMImages)):
            # find mapping to Dicom space  
            [transformed_image, transform_cube] = Display().dicomTransform(DICOMImages[i], image_pos_pat, image_ori_pat)  
            if (i==0):
                # create mask from segmenation
                np_VOI_mask = self.createMaskfromMesh(VOI_mesh, transformed_image)
            
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
                    pixVals_margin.append(pixValx)
            
            # Now collect pixVals
            print "\n Saving %s" % 'Fmargin'+str(i)
            Fmargin['Fmargin'+str(i)] = pixVals_margin
            pixVals_margin = []
            
            # extract pixID at inside VOIPnt
            VOI_scalars = transformed_image.GetPointData().GetScalars()
            np_VOI_imagedata = vtk_to_numpy(VOI_scalars)     
            
            dims = transformed_image.GetDimensions()
            spacing = transformed_image.GetSpacing()
            np_VOI_imagedata = np_VOI_imagedata.reshape(dims[2], dims[1], dims[0]) 
            np_VOI_imagedata = np_VOI_imagedata.transpose(2,1,0)
            
            #################### HERE GET INTERNAL PIXELS IT AND MASK IT OUT
            VOI_imagedata = np_VOI_imagedata[nonzero(np_VOI_mask)]
    
            for j in range( len(VOI_imagedata) ):
                pixValx = VOI_imagedata[j]
                pixVals.append(pixValx)
                   
            # Now collect pixVals
            print "\n Saving %s" % 'F'+str(i)
            voxel_frameS['F'+str(i)] = pixVals   
            pixVals = []
            
        ##############################################################
        # Initialize features
        self.i_var = []; self.alln_F_r_i=[]; self.allmin_F_r_i=[]; self.allmax_F_r_i=[]; 
        self.allmean_F_r_i=[]; self.allvar_F_r_i=[]; self.allskew_F_r_i=[]; self.allkurt_F_r_i=[]
        F_r_0 =  array(voxel_frameS['F'+str(0)]).astype(float)
        n, min_max, meanFr, var_F_r_0, skew, kurt = stats.describe(F_r_0)
        self.i_var_max = 0
                    
        # Collect to Compute inhomogeneity variance of uptake and other variables
        for k in range(1,len(DICOMImages)):
            F_r_i =  array(voxel_frameS['F'+str(k)]).astype(float)
            print "\nF_r_i parameters %s" % str(k)
            n_F_r_i, min_max_F_r_i, mean_F_r_i, var_F_r_i, skew_F_r_i, kurt_F_r_i = stats.describe(F_r_i)
                    
            print("Number of internal voxels: {0:d}".format(n_F_r_i))
            self.alln_F_r_i.append(n_F_r_i)
            print("Minimum: {0:8.6f} Maximum: {1:8.6f}".format(min_max_F_r_i[0], min_max_F_r_i[1]))
            self.allmin_F_r_i.append(min_max_F_r_i[0])
            self.allmax_F_r_i.append(min_max_F_r_i[1])
            print("Mean: {0:8.6f}".format(mean_F_r_i))
            self.allmean_F_r_i.append(mean_F_r_i)
            print("Variance F_r_i: {0:8.6f}".format(var_F_r_i))
            self.allvar_F_r_i.append(var_F_r_i)
            print("Skew : {0:8.6f}".format(skew_F_r_i))
            self.allskew_F_r_i.append(skew_F_r_i)
            print("Kurtosis: {0:8.6f}".format(kurt_F_r_i))
            self.allkurt_F_r_i.append(kurt_F_r_i)
            
            print("Variance of uptake: {0:8.6f}".format(var_F_r_i/var_F_r_0))                
            self.i_var.append( var_F_r_i/var_F_r_0 )
        
            # Find max of change in variance of uptake
            if( self.i_var[k-1] > self.i_var_max):
                self.i_var_max = self.i_var[k-1]        
    
        print("\nMax Variance of uptake: {0:8.6f}\n".format( self.i_var_max ))
        
        # Collect to Compute change in variance of uptake    
        self.ii_var = []
        self.ii_var_min = 1000
        for k in range(len(DICOMImages)-1):
            F_r_i =  array(voxel_frameS['F'+str(k)]).astype(float)
            F_r_iplus =  array(voxel_frameS['F'+str(k+1)]).astype(float)
            n, min_max, meanFr, var_F_r_ith, skew, kurt = stats.describe(F_r_i)
            n, min_max, meanFr, var_F_r_iplus, skew, kurt = stats.describe(F_r_iplus)
            
            """change Variance of uptake:"""
            self.ii_var.append( var_F_r_ith/var_F_r_iplus  )
        
            # Find max of change in variance of uptake
            if( var_F_r_ith/var_F_r_iplus < self.ii_var_min):
                self.ii_var_min = var_F_r_ith/var_F_r_iplus
        
        print("Min change Variance of uptake: {0:8.6f}\n".format( self.ii_var_min ))
        
        # Extract features for sharpness of lesion margin, compute Margin gradient iii_var
        # The gradient is computed using convolution with a 3D sobel filter using scipy.ndimage.filters.sobel
        # The function generic_gradient_magnitude calculates a gradient magnitude using the function passed through derivative to calculate first derivatives. 
        F_rmargin_0 =  array(Fmargin['Fmargin'+str(0)]).astype(float)
        self.iii_var_max = -1000
        iii_Sobelvar = []
        
        # Collect to Compute variance of uptake and other variables
        for k in range(1,len(DICOMImages)):    
            F_rmargin_i =  array(Fmargin['Fmargin'+str(k)]).astype(float)
            
            margin_delta = F_rmargin_i-F_rmargin_0
            # using first sobel and then prewitt
            sobel_grad_margin_delta = generic_gradient_magnitude(margin_delta, sobel) 
    
            # compute feature Margin Gradient
            n, min_max, mean_sobel_grad_margin, var, skew, kurt = stats.describe(sobel_grad_margin_delta)
            n, min_max, mean_F_rmargin_i, var_F_r_ith, skew, kurt = stats.describe(F_rmargin_i)
            
            """Margin Gradient"""
            iii_Sobelvar.append( mean_sobel_grad_margin/mean_F_rmargin_i )
        
            # Find max of Margin Gradient
            if( iii_Sobelvar[k-1] > self.iii_var_max):
                self.iii_var_max = iii_Sobelvar[k-1]
                self.iii_var_max_k = k
        
        print("Max Margin Gradient: {0:8.6f}".format( self.iii_var_max ))
        print("k for Max Margin Gradient: {0:8.6f}".format( self.iii_var_max_k ))
        
        # compute iv feature Variance of Margin Gradient
        # note: only computed from the subtraction frames of i and 0 where the margin gradient iii_var is maximum.
        self.ivVariance = []
        F_rmargin_iv =  array(Fmargin['Fmargin'+str(self.iii_var_max_k)]).astype(float)
        n, min_max, mean_F_rmargin_iv, var_F_r_ith, skew, kurt = stats.describe(F_rmargin_iv)
    
        margin_delta_iv = F_rmargin_iv-F_rmargin_0
        
        # using first sobel and then prewitt
        sobel_grad_margin_delta_iv = generic_gradient_magnitude(margin_delta_iv, sobel)
        n, min_max, mean_sobel, var_sobel_grad_margin_delta_iv, skew, kurt = stats.describe(sobel_grad_margin_delta_iv)
        
        self.ivVariance = var_sobel_grad_margin_delta_iv/mean_F_rmargin_iv**2
        
        print("Variance of spatial Margin Gradient: {0:8.6f}".format( self.ivVariance ))
        
        # Extract Shape features: pre-requisite is the Volume and the diameter of the lesion 
        ####################################
        # Measure VOI
        ###################################
        VOI_massProperty = vtk.vtkMassProperties()
        VOI_massProperty.SetInput(VOI_mesh)
        VOI_massProperty.Update()
        
        # get VOI volume
        # VTK is unitless. The units you get out are the units you put in.
        # If your input polydata has points defined in terms of millimetres, then
        # the volume will be in cubic millimetres. 
        VOI_vol = VOI_massProperty.GetVolume() # mm3
        VOI_surface = VOI_massProperty.GetSurfaceArea() # mm2
    
        # just print the results
        print "\nVolume lesion = ", VOI_vol
        print "Surface lesion  = ", VOI_surface
        
        # Calculate the effective diameter of the surface D=2(sqrt3(3V/(4pi))) 
        diam_root = (3*VOI_vol)/(4*pi)
        self.VOI_efect_diameter = 2*pow(diam_root,1.0/3) 
        print "VOI_efect_diameter = ", self.VOI_efect_diameter
            
        centerOfMassFilter = vtk.vtkCenterOfMass()
        centerOfMassFilter.SetInput( VOI_mesh )
        centerOfMassFilter.SetUseScalarsAsWeights(False)
        centerOfMassFilter.Update()
        
        # centroid of lesion 
        self.lesion_centroid = [0,0,0]
        self.lesion_centroid = centerOfMassFilter.GetCenter()
        print "lesion_centroid = ", self.lesion_centroid
        
        # create a sphere to compute the volume of lesion within a sphere of effective diameter
        sphere_effectD = vtk.vtkSphereSource()
        sphere_effectD.SetRadius(self.VOI_efect_diameter/2) #VOI_diameter/2
        sphere_effectD.SetCenter(self.lesion_centroid)
        sphere_effectD.Update()
        
        # compute volume of lesion within a sphere of effective diameter
        sphereVOI_massProperty = vtk.vtkMassProperties()
        sphereVOI_massProperty.SetInput(sphere_effectD.GetOutput())
        sphereVOI_massProperty.Update()
        sphereVOI_vol = sphereVOI_massProperty.GetVolume() # mm3
    
        # just print the results
        print "Volume sphereVOI = ", sphereVOI_vol
        
        # Compute Shape of lesion in 3D
        # Circularity
        epsilon = 0.001
        self.circularity = sphereVOI_vol/(VOI_vol+epsilon)
        print("\nCircularity: {0:8.6f}".format( self.circularity ))
        
        self.irregularity = 1 - pi*(self.VOI_efect_diameter/VOI_surface)
        print("Irregularity: {0:8.6f}".format( self.irregularity ))
        
        ####################################
        # Radial gradient analysis ref[9] white paper
        ###################################
        # Radial gradient analysis is based on examination of the angles between voxel-value gradients
        # and lines intersecting a single point near the center of the suspect lesion, lines in radial directions. 
        # Radial gradient values are given by the dot product of the gradient direction and the radial direction.
        RGH_mean = []; self.max_RGH_mean = 0; self.max_RGH_mean_k = 0; RGH_var = []; self.max_RGH_var = 0; self.max_RGH_var_k = 0;
        H_norm_p = []
        
        # do subtraction of timepost-pre
        #################### 
        for i in range(1,len(DICOMImages)):
            subtractedImage = Display().subImage(DICOMImages, i)
            [transformed_image, transform_cube] = Display().dicomTransform(subtractedImage, image_pos_pat, image_ori_pat)       
            
            for j in range( VOI_mesh.GetNumberOfPoints() ):
                VOI_mesh.GetPoint(j, VOIPnt)
                
                r = array(VOIPnt)
                rc = array(self.lesion_centroid)
                norm_rdir = (r-rc)/linalg.norm(r-rc)
               
                # Find point for gradient vectors at the margin point
                pixId = transformed_image.FindPoint(VOIPnt[0], VOIPnt[1], VOIPnt[2])
                sub_pt = [0,0,0]            
                transformed_image.GetPoint(pixId, sub_pt)
                
                ijk = [0,0,0]
                pco = [0,0,0]
                
                grad_pt = [0,0,0];
                
                inorout = transformed_image.ComputeStructuredCoordinates( sub_pt, ijk, pco)
                if(inorout == 0):
                    print "point outside data"
                else:
                    transformed_image.GetPointGradient( ijk[0], ijk[1], ijk[2], transformed_image.GetPointData().GetScalars(), grad_pt)
                    
                #############
                # Compute vector in the direction gradient at margin point
                grad_marginpt = array([grad_pt])
                norm_grad_marginpt = grad_marginpt/linalg.norm(grad_marginpt)
                
                # Compute dot product (unit vector for dot product)
                p_dot = dot(norm_grad_marginpt, norm_rdir)
                norm_p_dot = np.abs(p_dot)[0] #linalg.norm(p_dot)
                
                H_norm_p.append(norm_p_dot)    
            
            # The histogram of radial gradient values quantifying the frequency of occurrence of the dot products in a given region of interest
            # radial gradient histogram. The hist() function now has a lot more options
            # first create a single histogram
                        
            # the histogram of the data with histtype='step'
#            plt.figure()
#            nsamples, bins, patches = plt.hist(array(H_norm_p), 50, normed=1, histtype='bar',facecolor='blue', alpha=0.75)
#           n, min_max, mean_bins, var_bins, skew, kurt = stats.describe(nsamples)
            
            mean_bins = np.mean(H_norm_p)
            var_bins = np.var(H_norm_p)
            
            print("\n mean RGB: {0:8.6f}".format( mean_bins ))
            print("variance RGB: {0:8.6f}".format( var_bins ))
            
            # Append data
            RGH_mean.append( mean_bins )
            RGH_var.append( var_bins )
            
            # Find max of RGH Gradient
            if( RGH_mean[i-1] > self.max_RGH_mean):
                self.max_RGH_mean = RGH_mean[i-1]
                self.max_RGH_mean_k = i

            if( RGH_var[i-1] > self.max_RGH_var):
                self.max_RGH_var = RGH_var[i-1]
                self.max_RGH_var_k = i
                
            # add a line showing the expected distribution
            # create a histogram by providing the bin edges (unequally spaced)
            plt.xlabel('normalized dot product |R.G|')
            plt.ylabel('Probability')
            plt.title('radial gradient histogram')
            plt.grid(True)    
            
        ################# Jacob's lesion margin sharpness
        #initializations
        VOI_outlinept_normal = [0,0,0]; VOI_outlinept = [0,0,0];  inpt = [0,0,0];  outpt = [0,0,0]
        im_pts = [0,0,0]; ijk_in = [0,0,0]; ijk_out = [0,0,0]; pco = [0,0,0]; SIout_pixVal=[];	lastSIout_pixVal=[]
        
        # get model_point_normals
        VOI_point_normals = vtk.vtkPolyDataNormals()
        VOI_point_normals.SetInput( VOI_mesh )
        VOI_point_normals.SetComputePointNormals(1)
        VOI_point_normals.SetComputeCellNormals(0)
        VOI_point_normals.SplittingOff()
        VOI_point_normals.FlipNormalsOff()
        VOI_point_normals.ConsistencyOn()
        VOI_point_normals.Update()
        
        # Retrieve model normals
        VOI_normalsRetrieved = VOI_point_normals.GetOutput().GetPointData().GetNormals()
        VOI_n = VOI_normalsRetrieved.GetNumberOfTuples()
        
        # obtain vols of interest
        [transf_pre_dicomReader, transform_cube] = Display().dicomTransform(DICOMImages[0], image_pos_pat, image_ori_pat)
        [transf_last_dicomReader, transform_cube] = Display().dicomTransform(DICOMImages[len(DICOMImages)-1], image_pos_pat, image_ori_pat)
        
        num_margin = []
        den_margin = []   
        
        for i in range(1,len(DICOMImages)):
            #initializations
            SIout_pixVal=[]
            lastSIout_pixVal=[]
            
            subtractedImage = Display().subImage(DICOMImages, i)
            [transf_sub_pre_dicomReader, transform_cube] = Display().dicomTransform(subtractedImage, image_pos_pat, image_ori_pat) 
            
            
            for k in range( VOI_n ):
                VOI_outlinept_normal = VOI_normalsRetrieved.GetTuple3(k)
                VOI_mesh.GetPoint(k, VOI_outlinept)
                
                #   "d for radial lenght: %f" % d
                d = sqrt(spacing[0]**2 + spacing[1]**2 + spacing[2]**2)
                               
                inpt[0] = VOI_outlinept[0] - VOI_outlinept_normal[0]*d
                inpt[1] = VOI_outlinept[1] - VOI_outlinept_normal[1]*d
                inpt[2] = VOI_outlinept[2] - VOI_outlinept_normal[2]*d
        
                outpt[0] = VOI_outlinept[0] + VOI_outlinept_normal[0]*d
                outpt[1] = VOI_outlinept[1] + VOI_outlinept_normal[1]*d
                outpt[2] = VOI_outlinept[2] + VOI_outlinept_normal[2]*d
                
                # get pre-contrast SIin to normalized RSIgroup [See equation 1] from paper
                prepixin = transf_pre_dicomReader.FindPoint(inpt[0], inpt[1], inpt[2])
                transf_pre_dicomReader.GetPoint(prepixin,im_pts)
                transf_pre_dicomReader.ComputeStructuredCoordinates( im_pts, ijk_in, pco)
                #print ijk_in
                
                # get pre-contrast SIout in 6-c-neighbordhood to normalized RSIgroup [See equation 1] from paper
                prepixout = transf_pre_dicomReader.FindPoint(outpt[0], outpt[1], outpt[2])
                transf_pre_dicomReader.GetPoint(prepixout,im_pts)
                transf_pre_dicomReader.ComputeStructuredCoordinates( im_pts, ijk_out, pco)
                #print ijk_out
                
                # get t-post SIin
                SIin_pixVal = transf_sub_pre_dicomReader.GetScalarComponentAsFloat( ijk_in[0], ijk_in[1], ijk_in[2], 0 )
                preSIin_pixVal = transf_pre_dicomReader.GetScalarComponentAsFloat( ijk_in[0], ijk_in[1], ijk_in[2], 0 )+epsilon
    
                RSIin = SIin_pixVal/preSIin_pixVal
                ####            
                
                # get t-post SIout  6-c-neighbordhood
                #cn1
                SIout = transf_sub_pre_dicomReader.GetScalarComponentAsFloat( ijk_out[0]+1, ijk_out[1], ijk_out[2], 0 )
                preSIout = transf_pre_dicomReader.GetScalarComponentAsFloat( ijk_out[0]+1, ijk_out[1], ijk_out[2], 0 )+epsilon
                SIout_pixVal.append(float(SIout/preSIout))
                #cn2
                SIout = transf_sub_pre_dicomReader.GetScalarComponentAsFloat( ijk_out[0]-1, ijk_out[1], ijk_out[2], 0 )
                preSIout = transf_pre_dicomReader.GetScalarComponentAsFloat( ijk_out[0]-1, ijk_out[1], ijk_out[2], 0 )+epsilon
                SIout_pixVal.append(float(SIout/preSIout))
                #cn3
                SIout = transf_sub_pre_dicomReader.GetScalarComponentAsFloat( ijk_out[0], ijk_out[1]+1, ijk_out[2], 0 )
                preSIout = transf_pre_dicomReader.GetScalarComponentAsFloat( ijk_out[0], ijk_out[1]+1, ijk_out[2], 0 )+epsilon
                SIout_pixVal.append(float(SIout/preSIout))
                #cn4
                SIout = transf_sub_pre_dicomReader.GetScalarComponentAsFloat( ijk_out[0], ijk_out[1]-1, ijk_out[2], 0 )
                preSIout = transf_pre_dicomReader.GetScalarComponentAsFloat( ijk_out[0], ijk_out[1]-1, ijk_out[2], 0 )+epsilon
                SIout_pixVal.append(float(SIout/preSIout))
                #cn5
                SIout = transf_sub_pre_dicomReader.GetScalarComponentAsFloat( ijk_out[0], ijk_out[1], ijk_out[2]+1, 0 )
                preSIout = transf_pre_dicomReader.GetScalarComponentAsFloat( ijk_out[0], ijk_out[1], ijk_out[2]+1, 0 )+epsilon
                SIout_pixVal.append(float(SIout/preSIout))
                #cn6
                SIout = transf_sub_pre_dicomReader.GetScalarComponentAsFloat( ijk_out[0], ijk_out[1]-1, ijk_out[2]-1, 0 )
                preSIout = transf_pre_dicomReader.GetScalarComponentAsFloat( ijk_out[0], ijk_out[1]-1, ijk_out[2]-1, 0 )+epsilon
                SIout_pixVal.append(float(SIout/preSIout))
                
                RSIout = mean( SIout_pixVal ) 
                ###
                
                # get last-post SIout 6-c-neighbordhood
                #cn1
                SIout = transf_last_dicomReader.GetScalarComponentAsFloat( ijk_out[0]+1, ijk_out[1], ijk_out[2], 0 )
                preSIout = transf_pre_dicomReader.GetScalarComponentAsFloat( ijk_out[0]+1, ijk_out[1], ijk_out[2], 0 )+epsilon
                lastSIout_pixVal.append(float(SIout/preSIout))
                #cn2
                SIout = transf_last_dicomReader.GetScalarComponentAsFloat( ijk_out[0]-1, ijk_out[1], ijk_out[2], 0 )
                preSIout = transf_pre_dicomReader.GetScalarComponentAsFloat( ijk_out[0]-1, ijk_out[1], ijk_out[2], 0 )+epsilon
                lastSIout_pixVal.append(float(SIout/preSIout))
                #cn3
                SIout = transf_last_dicomReader.GetScalarComponentAsFloat( ijk_out[0], ijk_out[1]+1, ijk_out[2], 0 )
                preSIout = transf_pre_dicomReader.GetScalarComponentAsFloat( ijk_out[0], ijk_out[1]+1, ijk_out[2], 0 )+epsilon
                lastSIout_pixVal.append(float(SIout/preSIout))
                #cn4
                SIout = transf_last_dicomReader.GetScalarComponentAsFloat( ijk_out[0], ijk_out[1]-1, ijk_out[2], 0 )
                preSIout = transf_pre_dicomReader.GetScalarComponentAsFloat( ijk_out[0], ijk_out[1]-1, ijk_out[2], 0 )+epsilon
                lastSIout_pixVal.append(float(SIout/preSIout))
                #cn5
                SIout = transf_last_dicomReader.GetScalarComponentAsFloat( ijk_out[0], ijk_out[1], ijk_out[2]+1, 0 )
                preSIout = transf_pre_dicomReader.GetScalarComponentAsFloat( ijk_out[0], ijk_out[1], ijk_out[2]+1, 0 )+epsilon
                lastSIout_pixVal.append(float(SIout/preSIout))
                #cn6
                SIout = transf_last_dicomReader.GetScalarComponentAsFloat( ijk_out[0], ijk_out[1]-1, ijk_out[2]-1, 0 )
                preSIout = transf_pre_dicomReader.GetScalarComponentAsFloat( ijk_out[0], ijk_out[1]-1, ijk_out[2]-1, 0 )+epsilon
                lastSIout_pixVal.append(float(SIout/preSIout))

                # calculate
                RSIoutf = mean( lastSIout_pixVal )
                
                ### compute feature
                num_margin.append( RSIin-RSIout )
                den_margin.append( RSIin-RSIoutf )
                #print num_margin
                #print den_margin
                
                SIout_pixVal=[]
                lastSIout_pixVal=[]
             
        self.edge_sharp_mean = mean(array(num_margin).astype(float)) / mean(array(den_margin).astype(float))
        self.edge_sharp_std = std(array(num_margin).astype(float)) / std(array(den_margin).astype(float))
        print "\n edge_sharp_mean: "
        print self.edge_sharp_mean
                
        print "\n edge_sharp_std: "
        print self.edge_sharp_std
        
        ##################################################
        # orgamize into dataframe
        self.morphologyFeatures = DataFrame( data=array([[ mean(self.allmin_F_r_i), mean(self.allmax_F_r_i),mean(self.allmean_F_r_i), mean(self.allvar_F_r_i), mean(self.allskew_F_r_i), mean(self.allkurt_F_r_i),
            self.i_var_max, self.ii_var_min, self.iii_var_max, self.iii_var_max_k, self.ivVariance, self.circularity, self.irregularity, self.edge_sharp_mean, self.edge_sharp_std, self.max_RGH_mean, self.max_RGH_mean_k, self.max_RGH_var, self.max_RGH_var_k ]]), 
            columns=['min_F_r_i', 'max_F_r_i', 'mean_F_r_i', 'var_F_r_i', 'skew_F_r_i', 'kurt_F_r_i', 
            'iMax_Variance_uptake', 'iiMin_change_Variance_uptake', 'iiiMax_Margin_Gradient', 'k_Max_Margin_Grad', 'ivVariance', 'circularity', 'irregularity', 'edge_sharp_mean', 'edge_sharp_std', 'max_RGH_mean', 'max_RGH_mean_k', 'max_RGH_var', 'max_RGH_var_k'])
        
            
        return self.morphologyFeatures