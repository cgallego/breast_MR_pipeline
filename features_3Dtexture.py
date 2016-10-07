# -*- coding: utf-8 -*-
"""

Documentation: http://adessowiki.fee.unicamp.br/adesso/wiki/iatexture/glcmdesc/view/

Created on Wed Apr 02 09:40:11 2014
@ author (C) Cristina Gallego, University of Toronto
----------------------------------------------------------------------
"""


from numpy import *
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from pandas import DataFrame
from pylab import *
import matplotlib.pyplot as plt
from scipy import stats
from skimage.feature import greycomatrix, greycoprops
from graycomatrix3D import glcmdesc

from display import *

#!/usr/bin/env python
class Texture3D(object):
    """
    USAGE:
    =============
    features_3Dtexture = Texture3D()
    """
    def __init__(self):
        """ initialize Texture """
        self.texture_features = []


    def __call__(self):
        """ Turn Class into a callable object """
        Texture()

    def histeq(self, im, nbr_bins=256):
        #get image histogram
        imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)
        cdf = imhist.cumsum() #cumulative distribution function
        cdf = 255 * cdf / cdf[-1] #normalize

        #use linear interpolation of cdf to find new pixel values
        im2 = interp(im.flatten(),bins[:-1],cdf)

        return im2.reshape(im.shape), cdf


    def extractfeatures(self, DICOMImages, image_pos_pat, image_ori_pat, series_path, phases_series, VOI_mesh, VOI_efect_diameter, lesion_centroid, patchdirname):
        ###################################
        # Haralick et al. defined 10 grey-level co-occurrence matrix (GLCM) enhancement features (energy, maximum probability, contrast, homogeneity, entropy, correlation, sum average, sum variance, difference average and difference variance) to describe texture
        # N is the number of distinct gray levels in the histogram equalized image;
        allglcmdesc = []    
        for postt in range(1,len(DICOMImages)):
            # obtain vols of interest
            subtractedImage = Display().subImage(DICOMImages, postt)
            [sub_pre_transformed_image, transform_cube] = Display().dicomTransform(subtractedImage, image_pos_pat, image_ori_pat)

            print "\nLoading VOI_mesh MASK for Series %s... " % str(postt)
            VOIPnt = [0,0,0]; pixVals_margin = [];  pixVals = []

            #################### TO NUMPY
            VOI_scalars = sub_pre_transformed_image.GetPointData().GetScalars()
            np_VOI_imagedata = vtk_to_numpy(VOI_scalars)

            dims = sub_pre_transformed_image.GetDimensions()
            spacing = sub_pre_transformed_image.GetSpacing()
            np_VOI_imagedata = np_VOI_imagedata.reshape(dims[2], dims[1], dims[0])
            np_VOI_imagedata = array(np_VOI_imagedata.transpose(2,1,0))

             #****************************"
            print "Normalizing data..."
            np_VOI_imagedataflat = np_VOI_imagedata.flatten().astype(float)
            np_VOI_imagedata_num = (np_VOI_imagedataflat-min(np_VOI_imagedataflat))
            np_VOI_imagedata_den = (max(np_VOI_imagedataflat)-min(np_VOI_imagedataflat))
            np_VOI_imagedata_flatten = 255*(np_VOI_imagedata_num/np_VOI_imagedata_den)
            np_VOI_imagedata = np_VOI_imagedata_flatten.reshape(dims[0], dims[1], dims[2])
            
            lesion_patches = []
            
            # Prepare lesion localization and PATCH size for Texture analysis
            # lesion_centroid
            lesion_radius = VOI_efect_diameter/(spacing[0])
            lesionthick = VOI_efect_diameter/(spacing[2])
            print "VOI_efect_diameter %s... " % str(lesion_radius)
    
            lesion_location = lesion_centroid
            print "lesion_location %s... " % str(lesion_location)

            ######### translate lesion location to ijk coordinates
            # sub_pre_transformed_image.FindPoint(lesion_location
            pixId = sub_pre_transformed_image.FindPoint(lesion_location[0], lesion_location[1], lesion_location[2])
            sub_pt = [0,0,0]
            sub_pre_transformed_image.GetPoint(pixId, sub_pt)
            ijk = [0,0,0]
            pco = [0,0,0]

            inorout = sub_pre_transformed_image.ComputeStructuredCoordinates( sub_pt, ijk, pco)
            print "coresponding ijk_vtk indices:"
            print ijk
            ijk_vtk = ijk

            # Perform texture classification using grey level co-occurrence matrices (GLCMs).
            # A GLCM is a histogram of co-occurring greyscale values at a given offset over an image.
            # compute some GLCM properties each patch
            # p(i,j) is the (i,j)th entry in a normalized spatial gray-level dependence matrix;
            lesion_patches = []
            lesion_patches = np_VOI_imagedata[
                    int(ijk_vtk[0] - lesion_radius-1):int(ijk_vtk[0] + lesion_radius+1),
                    int(ijk_vtk[1] - lesion_radius-1):int(ijk_vtk[1] + lesion_radius),
                    int(ijk_vtk[2] - lesionthick-1):int(ijk_vtk[2] + lesionthick+1) ]

            print '\n Lesion_patches:'
            print lesion_patches

            #################### RESAMPLE TO ISOTROPIC
            # pass to vtk
            lesion_patches_shape = lesion_patches.shape
            vtklesion_patches = lesion_patches.transpose(2,1,0)
            vtklesion_patches_data = numpy_to_vtk(num_array=vtklesion_patches.ravel(), deep=True, array_type=vtk.VTK_FLOAT)

            # probe into the unstructured grid using ImageData geometry
            vtklesion_patches = vtk.vtkImageData()
            vtklesion_patches.SetExtent(0,lesion_patches_shape[0]-1,0,lesion_patches_shape[1]-1,0,lesion_patches_shape[2]-1)
            vtklesion_patches.SetOrigin(0,0,0)
            vtklesion_patches.SetSpacing(spacing)
            vtklesion_patches.GetPointData().SetScalars(vtklesion_patches_data)

            # write ####################
            isowriter = vtk.vtkMetaImageWriter()
            isowriter.SetInput(vtklesion_patches)
            isowriter.SetFileName(patchdirname+"_"+str(postt)+".mha")
            isowriter.Write()

            isopix = mean(vtklesion_patches.GetSpacing())
            resample = vtk.vtkImageResample ()
            resample.SetInput( vtklesion_patches )
            resample.SetAxisOutputSpacing( 0, isopix )
            resample.SetAxisOutputSpacing( 1, isopix )
            resample.SetAxisOutputSpacing( 2, isopix )
            resample.Update()
            isoImage = resample.GetOutput()

            #################### get isotropic patches
            ISO_scalars = isoImage.GetPointData().GetScalars()
            np_ISO_Image = vtk_to_numpy(ISO_scalars)

            isodims = isoImage.GetDimensions()
            isospacing = isoImage.GetSpacing()
            np_ISO_Imagedata = np_ISO_Image.reshape(isodims[2], isodims[1], isodims[0])
            np_ISO_Imagedata = array(np_ISO_Imagedata.transpose(2,1,0))

            #################### save isotropic patches
            fig = plt.figure()
            # patch histogram
            ax1 = fig.add_subplot(221)
            n, bins, patches = plt.hist(array(np_ISO_Imagedata.flatten()), 50, normed=1, facecolor='green', alpha=0.75)
            ax1.set_ylabel('histo')

            ax2 = fig.add_subplot(222)
            plt.imshow(np_ISO_Imagedata[:,:,np_ISO_Imagedata.shape[2]/2])
            plt.gray()
            ax2.set_ylabel('iso: '+str(isodims) )

            ax3 = fig.add_subplot(223)
            plt.imshow(lesion_patches[:,:,lesion_patches.shape[2]/2])
            plt.gray()
            ax3.set_ylabel('original: '+str(lesion_patches_shape))

            ax4 = fig.add_subplot(224)
            plt.imshow(np_VOI_imagedata[:,:,ijk_vtk[2]])
            plt.gray()
            ax4.set_ylabel('lesion centroid(ijk): '+str(ijk_vtk))

            # FInally display
            # plt.show()
            plt.savefig(patchdirname+"_"+str(postt)+'.png', format='png')
            ####################

            #################### 3D GLCM
            from graycomatrix3D import glcm3d

            patch = np_ISO_Imagedata.astype(np.uint8)
            lev = int(patch.max()+1) # levels
            
            # perfor glmc extraction in all 13 directions in 3D pixel neighbors
            g1 = glcm3d(lev, patch, offsets=[0,0,1])  # orientation 0 degrees (example same slices: equal to Nslices*0degree 2D case)
            g2 = glcm3d(lev, patch, offsets=[0,1,-1]) # orientation 45 degrees (example same slices: equal to Nslices*45degree 2D case)
            g3 = glcm3d(lev, patch, offsets=[0,1,0]) # orientation 90 degrees (example same slices: equal to Nslices*90degree 2D case)
            g4 = glcm3d(lev, patch, offsets=[0,1,1]) # orientation 135 degrees (example same slices: equal to Nslices*135degree 2D case)
            g5 = glcm3d(lev, patch, offsets=[1,0,-1]) # 0 degrees/45 degrees (example same slices: equal to (Nslices-1)*0degree 2D case)
            g6 = glcm3d(lev, patch, offsets=[1,0,0])  # straight up (example same slices: equal to np.unique())
            g7 = glcm3d(lev, patch, offsets=[1,0,1]) # 0 degree/135 degrees (example same slices: equal to (Nslices-1)*transpose of 0degree 2D case)
            g8 = glcm3d(lev, patch, offsets=[1,1,0]) # 90 degrees/45 degrees (example same slices: equal to (Nslices-1)*90 degree 2D case)
            g9 = glcm3d(lev, patch, offsets=[1,-1,0])    # 90 degrees/135 degrees (example same slices: equal to (Nslices-1)*transpose of 90 degree 2D case)
            g10 = glcm3d(lev, patch, offsets=[1,1,-1])    # 45 degrees/45 degrees (example same slices: equal to (Nslices-1)*45 degree 2D case)
            g11 = glcm3d(lev, patch, offsets=[1,-1,1])   # 45 degree/135 degrees (example same slices: equal to (Nslices-1)*transpose of 45 degree 2D case)
            g12 = glcm3d(lev, patch, offsets=[1,1,1])    # 135 degrees/45 degrees (example same slices: equal to (Nslices-1)*135 degree 2D case)
            g13 = glcm3d(lev, patch, offsets=[0,0,1])    # 135 degrees/135 degrees (example same slices: equal to (Nslices-1)*transpose of 135 degree 2D case)

            # plot                
            fig = plt.figure()
            fig.add_subplot(431);           plt.imshow(g1); plt.gray()
            fig.add_subplot(432);           plt.imshow(g2); plt.gray()
            fig.add_subplot(433);           plt.imshow(g3); plt.gray()
            fig.add_subplot(434);           plt.imshow(g4); plt.gray()
            fig.add_subplot(435);           plt.imshow(g5); plt.gray()
            fig.add_subplot(436);           plt.imshow(g6); plt.gray()
            fig.add_subplot(437);           plt.imshow(g7); plt.gray()
            fig.add_subplot(438);           plt.imshow(g8); plt.gray()
        
            # add all directions to make features non-directional
            g = g1+g2+g3+g4+g5+g6+g7+g8+g9+g10+g11+g12+g13
        
            fig.add_subplot(439);           plt.imshow(g); plt.gray()
            #plt.show()
            
            ### glcm normalization ###
            if g.sum() != 0:
                g = g.astype(float)/g.sum()
            
            ### compute auxiliary variables ###
            (num_level, num_level2) = g.shape
            I, J = np.ogrid[0:num_level, 0:num_level]
            I = 1+ np.array(range(num_level)).reshape((num_level, 1))
            J = 1+ np.array(range(num_level)).reshape((1, num_level))
            diff_i = I - np.apply_over_axes(np.sum, (I * g), axes=(0, 1))[0, 0]
            diff_j = J - np.apply_over_axes(np.sum, (J * g), axes=(0, 1))[0, 0]
            std_i = np.sqrt(np.apply_over_axes(np.sum, (g * (diff_i) ** 2),axes=(0, 1))[0, 0])
            std_j = np.sqrt(np.apply_over_axes(np.sum, (g * (diff_j) ** 2),axes=(0, 1))[0, 0])
            cov = np.apply_over_axes(np.sum, (g * (diff_i * diff_j)),axes=(0, 1))[0, 0]
            
            gxy = np.zeros(2*g.shape[0]-1)   ### g x+y
            gx_y = np.zeros(g.shape[0])  ### g x-y
            for i in xrange(g.shape[0]):
                for j in xrange(g.shape[0]):
                    gxy[i+j] += g[i,j]
                    gx_y[np.abs(i-j)] += g[i,j]
            
            mx_y = (gx_y*np.arange(len(gx_y))).sum()
            v = np.zeros(11)
            i,j = np.indices(g.shape)+1
            ii = np.arange(len(gxy))+2
            ii_ = np.arange(len(gx_y))
            
            ### compute descriptors ###
            v[0] = np.apply_over_axes(np.sum, (g ** 2), axes=(0, 1))[0, 0] # energy or Angular second moment
            v[1] = np.apply_over_axes(np.sum, (g * ((I - J) ** 2)), axes=(0, 1))[0, 0] # Contrast
            if std_i>1e-15 and std_j>1e-15: # handle the special case of standard deviations near zero
                v[2] = cov/(std_i*std_j)#v[2] = greycoprops(g,'correlation') # Correlation
            else:
                v[2] = 1
            v[3] = np.apply_over_axes(np.sum, (g* (diff_i) ** 2),axes=(0, 1))[0, 0]# Variance or Sum of squares
            v[4] = np.sum(g * (1. / (1. + (I - J) ** 2))) # Inverse difference moment
            v[5] = (gxy*ii).sum() # Sum average
            v[6] = ((ii-v[5])*(ii-v[5])*gxy).sum() # Sum variance
            v[7] = -1*(gxy*np.log10(gxy+ np.spacing(1))).sum() # Sum entropy
            v[8] = -1*(g*np.log10(g+np.spacing(1))).sum() # Entropy
            v[9] = ((ii_-mx_y)*(ii_-mx_y)*gx_y).sum() # Difference variance
            v[10] = -1*(gx_y*np.log10(gx_y++np.spacing(1))).sum() # Difference entropy
            
            """TEXTURE FEATURES accumulated"""
            allglcmdesc = append(allglcmdesc, v)
            
            
        ##################################################
        # writing to file from row_lesionID Drow_PathRepID
        print "\n Append texture features for each post contrast"
        [self.texture_energy_nondir_post1, self.texture_contrast_nondir_post1, self.texture_correlation_nondir_post1,
         self.texture_variance_nondir_post1, self.texture_inversediffmoment_nondir_post1, self.texture_sumaverage_nondir_post1,
         self.texture_sumvariance_nondir_post1, self.texture_sumentropy_nondir_post1, self.texture_entropy_nondir_post1,
         self.texture_diffvariance_nondir_post1,self.texture_diffentropy_nondir_post1] = allglcmdesc[0:11]
         
        [self.texture_energy_nondir_post2, self.texture_contrast_nondir_post2, self.texture_correlation_nondir_post2, 
         self.texture_variance_nondir_post2, self.texture_inversediffmoment_nondir_post2, self.texture_sumaverage_nondir_post2,
         self.texture_sumvariance_nondir_post2, self.texture_sumentropy_nondir_post2, self.texture_entropy_nondir_post2, 
         self.texture_diffvariance_nondir_post2, self.texture_diffentropy_nondir_post2] = allglcmdesc[11:22] 
         
        [self.texture_energy_nondir_post3, self.texture_contrast_nondir_post3, self.texture_correlation_nondir_post3, 
         self.texture_variance_nondir_post3, self.texture_inversediffmoment_nondir_post3, self.texture_sumaverage_nondir_post3,
         self.texture_sumvariance_nondir_post3, self.texture_sumentropy_nondir_post3, self.texture_entropy_nondir_post3, 
         self.texture_diffvariance_nondir_post3, self.texture_diffentropy_nondir_post3] = allglcmdesc[22:33] 
         
        [self.texture_energy_nondir_post4, self.texture_contrast_nondir_post4, self.texture_correlation_nondir_post4, 
         self.texture_variance_nondir_post4, self.texture_inversediffmoment_nondir_post4, self.texture_sumaverage_nondir_post4, 
         self.texture_sumvariance_nondir_post4, self.texture_sumentropy_nondir_post4, self.texture_entropy_nondir_post4, 
         self.texture_diffvariance_nondir_post4, self.texture_diffentropy_nondir_post4] = allglcmdesc[33:44] 

        # orgamize into dataframe
        self.texture_features = DataFrame( data=array([[self.texture_energy_nondir_post1, self.texture_contrast_nondir_post1, self.texture_correlation_nondir_post1,
                                                        self.texture_variance_nondir_post1, self.texture_inversediffmoment_nondir_post1, self.texture_sumaverage_nondir_post1,
                                                        self.texture_sumvariance_nondir_post1, self.texture_sumentropy_nondir_post1, self.texture_entropy_nondir_post1,
                                                        self.texture_diffvariance_nondir_post1,self.texture_diffentropy_nondir_post1,
                                                        self.texture_energy_nondir_post2, self.texture_contrast_nondir_post2, self.texture_correlation_nondir_post2, 
                                                        self.texture_variance_nondir_post2, self.texture_inversediffmoment_nondir_post2, self.texture_sumaverage_nondir_post2,
                                                        self.texture_sumvariance_nondir_post2, self.texture_sumentropy_nondir_post2, self.texture_entropy_nondir_post2, 
                                                        self.texture_diffvariance_nondir_post2, self.texture_diffentropy_nondir_post2, 
                                                        self.texture_energy_nondir_post3, self.texture_contrast_nondir_post3, self.texture_correlation_nondir_post3, 
                                                        self.texture_variance_nondir_post3, self.texture_inversediffmoment_nondir_post3, self.texture_sumaverage_nondir_post3,
                                                        self.texture_sumvariance_nondir_post3, self.texture_sumentropy_nondir_post3, self.texture_entropy_nondir_post3, 
                                                        self.texture_diffvariance_nondir_post3, self.texture_diffentropy_nondir_post3, 
                                                        self.texture_energy_nondir_post4, self.texture_contrast_nondir_post4, self.texture_correlation_nondir_post4, 
                                                        self.texture_variance_nondir_post4, self.texture_inversediffmoment_nondir_post4, self.texture_sumaverage_nondir_post4, 
                                                        self.texture_sumvariance_nondir_post4, self.texture_sumentropy_nondir_post4, self.texture_entropy_nondir_post4, 
                                                        self.texture_diffvariance_nondir_post4, self.texture_diffentropy_nondir_post4]]),
        columns=[    'texture_energy_nondir_post1','texture_contrast_nondir_post1','texture_correlation_nondir_post1', 'texture_variance_nondir_post1', 'texture_inversediffmoment_nondir_post1', 'texture_sumaverage_nondir_post1', 'texture_sumvariance_nondir_post1', 
                     'texture_sumentropy_nondir_post1', 'texture_entropy_nondir_post1', 'texture_diffvariance_nondir_post1', 'texture_diffentropy_nondir_post1', 'texture_energy_nondir_post2', 'texture_contrast_nondir_post2',  'texture_correlation_nondir_post2', 
                     'texture_variance_nondir_post2', 'texture_inversediffmoment_nondir_post2', 'texture_sumaverage_nondir_post2', 'texture_sumvariance_nondir_post2', 'texture_sumentropy_nondir_post2', 'texture_entropy_nondir_post2', 'texture_diffvariance_nondir_post2', 
                     'texture_diffentropy_nondir_post2', 'texture_energy_nondir_post3', 'texture_contrast_nondir_post3',  'texture_correlation_nondir_post3', 'texture_variance_nondir_post3', 'texture_inversediffmoment_nondir_post3', 'texture_sumaverage_nondir_post3', 
                     'texture_sumvariance_nondir_post3', 'texture_sumentropy_nondir_post3', 'texture_entropy_nondir_post3', 'texture_diffvariance_nondir_post3', 'texture_diffentropy_nondir_post3',  'texture_energy_nondir_post4', 'texture_contrast_nondir_post4',  
                     'texture_correlation_nondir_post4', 'texture_variance_nondir_post4', 'texture_inversediffmoment_nondir_post4', 'texture_sumaverage_nondir_post4', 'texture_sumvariance_nondir_post4', 'texture_sumentropy_nondir_post4', 'texture_entropy_nondir_post4', 
                     'texture_diffvariance_nondir_post4', 'texture_diffentropy_nondir_post4', ])


        return self.texture_features