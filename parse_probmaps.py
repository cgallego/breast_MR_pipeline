# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 13:42:08 2016

Code to run Hongbos list of processed files based on on Cristina's verified groud truth list:
    Z:\\Breast\\gtSeg_verified\\CADlesions_wSeries.csv'

Input files:
 - listofprobmaps.txt   (619 cases ~ 50/50 lesion/normals)
     - list of mhas of lesion probability maps in Y:\\Hongbo\\segmentations_train
     - to discover in Y:\\Hongbo\\processed_data:
         - mc Left/Right according to listofprobmaps.txt syntax: 1_0002_6745896_right_2_1_NS_NF_probmap.mha
 - listofprobmaps_test.txt  (450 cases ~ 33/66 lesion/normals)
     - list of mhas of lesion probability maps in Y:\\Hongbo\\segmentations_test
     - to discover in Y:\\Hongbo\\processed_data:
         - mc Left/Right according to listofprobmaps.txt syntax: 1_0002_6745896_right_2_1_NS_NF_probmap.mha

Info about lesions:
 - currently in database textureUpdatedFeatures.db segmentation.lesion_centroid_ijk will have centroid [ijk] coords
 - centroid [ijk] coords and gt mask Y:\Hongbo\gt_data are in bilateral coords, need to split in halfs Left/Right round(half)/rest
 - db segmentation.segm_xmin/max .segm_ymin/max .segm_zmin/max spans # pixels of lesion in each dim (lesion size)
 
Size considerations:
 - breast pixels inside probmap ~ 3/4 of entire volume. So total of:
     512*512*44*0.75 = 8million pixels/Volume 
     512*512*0.75 = 200K pixels/slice

@author: DeepLearning

"""

import pandas as pd
import os
import os.path, sys
import shutil
import glob
import numpy as np
import SimpleITK as sitk

# to query local databse
from mylocalbase import localengine
from sqlalchemy.orm import sessionmaker
import mylocaldatabase_new

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

import numpy.ma as ma
from skimage.measure import find_contours, approximate_polygon

from scipy.spatial import Delaunay
from matplotlib.collections import LineCollection

# to save graphs
import six.moves.cPickle as pickle
import gzip


def querylocalDatabase_wRad(lesion_id, verbose=False):
    """ Querying without known condition (e.g: mass, non-mass) if benign by assumption query only findings"""
    #############################
    ###### 1) Querying Research database for clinical, pathology, radiology data
    #############################
    print "Executing local connection..."
    # Create the database: the Session.
    Session = sessionmaker()
    Session.configure(bind=localengine)  # once engine is available
    session = Session() #instantiate a Session
       
    # perform query
    try:
        ############# by lesion id
        for lesion in session.query(mylocaldatabase_new.Lesion_record, mylocaldatabase_new.Radiology_record, mylocaldatabase_new.Segment_record).\
            filter(mylocaldatabase_new.Radiology_record.lesion_id==mylocaldatabase_new.Lesion_record.lesion_id).\
            filter(mylocaldatabase_new.Segment_record.lesion_id==mylocaldatabase_new.Lesion_record.lesion_id).\
            filter(mylocaldatabase_new.Lesion_record.lesion_id == str(lesion_id)):
            # print results
            if not lesion:
                print "lesion is empty"
            
        casesFrame = pd.Series(lesion.Lesion_record.__dict__)
        if verbose:
            casesFrame.to_csv(sys.stdout)            
            print(lesion)
        
    except Exception:
        return -1
           
    # if correctly proccess
    #slice data, get only 1 record        
    is_mass = list(lesion.Lesion_record.mass_lesion)
    if(is_mass):
        print "\n MASS:"
        cond = 'mass'
        mass = pd.Series(is_mass[0])
        mass_Case =  pd.Series(mass[0].__dict__)
        if verbose:
            mass_Case.to_csv(sys.stdout) 
        # decide if it's a mass or nonmass
        MorNMcase = mass_Case
        
    is_nonmass = list(lesion.Lesion_record.nonmass_lesion)
    if(is_nonmass):
        print "\n NON-MASS:"
        cond = 'nonmass'
        nonmass = pd.Series(is_nonmass[0])
        nonmass_Case =  pd.Series(nonmass[0].__dict__)
        if verbose:
            nonmass_Case.to_csv(sys.stdout) 
        # decide if it's a mass or nonmass
        MorNMcase = nonmass_Case
        
    is_foci = list(lesion.Lesion_record.foci_lesion)
    if(is_foci):
        print "\n FOCI:"
        cond = 'foci'
        foci = pd.Series(is_foci[0])
        foci_Case =  pd.Series(foci[0].__dict__)
        if verbose:
            foci_Case.to_csv(sys.stdout) 
        # decide if it's a mass or nonmass
        MorNMcase = foci_Case
        
    print "\n----------------------------------------------------------"          
    BenignNMaligNAnt = casesFrame['lesion_label'][-1]
    Diagnosis = casesFrame['lesion_diagnosis']
    print(Diagnosis)
    
    lesion_coords = pd.Series(lesion.Segment_record.__dict__)
    if verbose:
        print(lesion_coords)
                    
    return cond, BenignNMaligNAnt, Diagnosis, casesFrame, MorNMcase, lesion_coords
    

def plotSlice_outlines(probmap, breastm, zslice, ax, ploi, bg_img=None):
    """Subroutine to plot given a nrow, ncol position a single slice with combined outlines
    with the option to pass a bg_img to plot as imshow with triangulation overlay. default = None to plots probmap"""
    outlines_queryprob = find_contours(probmap[zslice,:,:], 0.615)
    if(outlines_queryprob):
        outlines_pts = np.concatenate(outlines_queryprob)
        
        # add breast mask outline
        outline_breastm = find_contours(breastm[zslice,:,:], 0)[0]
        coords_breastm = approximate_polygon(outline_breastm, tolerance=2)

        ax[ploi].plot(coords_breastm[:, 1], coords_breastm[:, 0], '.b', linewidth=0.5)
        ax[ploi].axis((0, probmap.shape[1], probmap.shape[2], 0))

        ax[ploi].imshow(probmap[zslice,:,:], cmap=plt.cm.gray)
        if(hasattr(bg_img, "__len__")):
            ax[ploi].imshow(bg_img[zslice,:,:], cmap=plt.cm.gray)
        ax[ploi].set_adjustable('box-forced')
        ax[ploi].plot(outlines_pts[:, 1], outlines_pts[:, 0], '.r', linewidth=0.5)
        
        # combine and simplify all
        outlines_all = np.concatenate([outlines_pts, coords_breastm])      
        coords_all = approximate_polygon(outlines_all, tolerance=2)
        ax[ploi].plot(coords_all[:, 1], coords_all[:, 0], '.g', linewidth=0.5)
        
        ################
        # Add Delaunay triangulation
        ################
        # perform Delaunay triangulation on the pts for query lesion
        a = np.asarray( [coords_all[:, 1], coords_all[:, 0]] ).transpose()
        lesion_delaunay = Delaunay(a)
        lesion_triangles = lesion_delaunay.points[lesion_delaunay.vertices]
        
        lines = []
        # Triangle vertices
        A = lesion_triangles[:, 0]
        B = lesion_triangles[:, 1]
        C = lesion_triangles[:, 2]
        lines.extend(zip(A, B))
        lines.extend(zip(B, C))
        lines.extend(zip(C, A))
        lines = LineCollection(lines, color='c')
        ax[ploi].add_collection(lines)   
        # turn axes off
        ax[ploi].get_xaxis().set_visible(False)
        ax[ploi].get_yaxis().set_visible(False)
    
    return 
    
def run_parseprobmaps():
    # Get Root folder ( the directory of the script being run)
    path_rootFolder = os.path.dirname(os.path.abspath(__file__))
    os.chdir(path_rootFolder) 
    
    processed_path = 'Y:\\Hongbo\\processed_data'
    probmaps_path = 'Y:\\Hongbo\\segmentations_train'
    gt_path = 'Y:\\Hongbo\\gt_data'
    output_path = 'Z:\\Cristina\\Section3\\breast_MR_pipeline\\graphs'
    parseprobmaps_path = 'Z:\\Cristina\\Section3\\breast_MR_pipeline\\parse_probmaps'
        
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(parseprobmaps_path):
        os.mkdir(parseprobmaps_path)
        
    file_ids = open("listofprobmaps.txt","r") 
    file_ids.seek(0)
    line = file_ids.readline()
  

    while ( line ) : 
        # Get the line: id  CAD study #	Accession #	Date of Exam	Pathology Diagnosis 
        print(line)
        fileline = line.split('_')
        lesion_id = int(fileline[0] )
        fStudyID = fileline[1] 
        AccessionN = fileline[2]  
        sideB = fileline[3]
    
        #############################
        ###### 1) Querying Research database for clinical, pathology, radiology data
        #############################
        cond, BenignNMaligNAnt, Diagnosis, casesFrame, MorNMcase, lesion_coords = querylocalDatabase_wRad(lesion_id, verbose=True)        
        
        #############################
        ###### 2) Accesing mc images, prob maps, gt_lesions and breast masks
        #############################
        # get dynmic series info
        DynSeries_id = MorNMcase['DynSeries_id']
        precontrast_id = int(str(DynSeries_id[1:])) #s600, [1:] remove the 's'
        DynSeries_nums = [str(n) for n in range(precontrast_id,precontrast_id+5)]

        DynSeries_imagefiles = []
        mriVols = []
        print "Reading MRI volumes..."
        for j in range(5):
            #the output mha:lesionid_patientid_access#_series#@acqusionTime.mha
            DynSeries_filename = '{}_{}_{}_{}'.format(str(lesion_id),fStudyID.zfill(4),AccessionN,DynSeries_nums[j] )
            
            #write log if mha file not exist             
            glob_result = glob.glob(os.path.join(processed_path,DynSeries_filename+'*')) #'*':do not to know the exactly acquistion time
            if glob_result != []:
                filename = glob_result[0]
                
            # add side info from the side of the lesion
            filename_wside = filename[:-4]+'_{}.mha'.format(sideB)
            DynSeries_imagefiles.append(filename_wside)
               
            # read Volumnes
            mriVols.append( sitk.GetArrayFromImage(sitk.Cast(sitk.ReadImage(DynSeries_imagefiles[j]),sitk.sitkFloat32)) )
        
        # read only one bilateral volume for left/right splitting
        bilateral_filename = '{}_{}_{}_{}@'.format(str(lesion_id),fStudyID.zfill(4),AccessionN,DynSeries_nums[1] )
        bilateral_filepath = glob.glob(os.path.join( processed_path,bilateral_filename+'*'+'_mc.mha' ))[0]
        bilateral_Vol = sitk.GetArrayFromImage(sitk.Cast(sitk.ReadImage(bilateral_filepath),sitk.sitkFloat32)) 
        # get original No of slices
        nslices = bilateral_Vol.shape[0]
        lefts = int(round(nslices/2))
        rights = nslices - lefts
        
        #get wofs filename
        print "Reading probability map..."
        probmap_filename = '{}_{}_{}_{}#2_1_NS_NF_probmap.mha'.format(str(lesion_id),fStudyID.zfill(4),AccessionN,sideB)
        probmap_filepath = os.path.join(probmaps_path,probmap_filename)
        probmap = sitk.GetArrayFromImage(sitk.Cast( sitk.ReadImage(probmap_filepath), sitk.sitkFloat32)) 
        
        print "Reading gt_lesions mask..."
        gt_lesion_filename = '{}_{}_{}.mha'.format(str(lesion_id),fStudyID.zfill(4),AccessionN)
        gt_lesion_filepath = os.path.join(gt_path,gt_lesion_filename)
        gt_lesion = sitk.GetArrayFromImage(sitk.Cast( sitk.ReadImage(gt_lesion_filepath), sitk.sitkFloat32)) 

        print "Reading breast mask..." 
        breastm_filename = '{}_{}_{}_wofs_reg_breastmask_{}.mha'.format(str(lesion_id),fStudyID.zfill(4),AccessionN,sideB)
        breastm_filepath = os.path.join(processed_path,breastm_filename)
        breastm = sitk.GetArrayFromImage(sitk.Cast( sitk.ReadImage(breastm_filepath), sitk.sitkFloat32))
        
        print "Pinpointing lesion slice..." 
        centroid_info = lesion_coords['lesion_centroid_ijk']
        centroid = [int(c) for c in centroid_info[1:-1].split(',')]
        
        #based onsplit left/right and recalculate centroid
        lesions = centroid[2]
        if(lesions > lefts): # lesion is on the right
            centroid[2] = centroid[2]-lefts
            # compute gt_lesion based on split left/right
            side_gt_lesion = gt_lesion[rights:,:,:]
        else:
            side_gt_lesion = gt_lesion[:lefts,:,:]
            
        # Display 
        fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(20, 10))
        for k in range(5):
            ax[0,k].imshow(mriVols[k][centroid[2],:,:], cmap=plt.cm.gray)
            ax[0,k].set_adjustable('box-forced')
            ax[0,k].set_xlabel(str(k)+'mc_'+sideB)
            
        # Display breastm 
        ax[1,0].imshow(breastm[centroid[2],:,:], cmap=plt.cm.gray)
        ax[1,0].set_adjustable('box-forced')
        ax[1,0].set_xlabel(sideB+'_breastm')
        
        # Display outline of breastm (using skimage)
        # Using the “marching squares” method to compute a the iso-valued contours 
        # of the input 2D array for a particular level value. 
        # Array values are linearly interpolated to provide better precision for the output contours.
        outline_breastm = find_contours(breastm[centroid[2],:,:], 0)[0]
        
        # Approximate a polygonal chain with the specified tolerance.
        # It is based on the Douglas-Peucker algorithm.
        # Note that the approximated polygon is always within the convex hull of the original polygon.
        # ref https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
        # The algorithm recursively divides the line. Initially it is given all the points between the first and last point. 
        # It automatically marks the first and last point to be kept. It then finds the point that is furthest from the 
        # line segment with the first and last points as end points; this point is obviously furthest on the curve from the
        # approximating line segment between the end points.
        # If the point is closer than ε to the line segment then any points not currently marked to be kept can be discarded 
        # without the simplified curve being worse than ε.
        # If the point furthest from the line segment is greater than ε from the approximation then that point must be kept. 
        # The algorithm recursively calls itself with the first point and the worst point and then with the worst point and 
        # the last point, which includes marking the worst point being marked as kept.
        # When the recursion is completed a new output curve can be generated consisting of all and only 
        # those points that have been marked as kept.
        coords_breastm = approximate_polygon(outline_breastm, tolerance=2)
        print "Number of coordinates: %d to %d = %d percent reduction" % ( len(outline_breastm), len(coords_breastm), float(len(outline_breastm)-len(coords_breastm))/len(outline_breastm)*100 )

        # Display gt_lesion based on split left/right
        ax[1,1].imshow(side_gt_lesion[centroid[2],:,:], cmap=plt.cm.gray)
        ax[1,1].set_adjustable('box-forced')
        ax[1,1].set_xlabel(sideB+'_gt_lesion')

        # Display probmap 
        ax[1,2].imshow(probmap[centroid[2],:,:], cmap=plt.cm.gray)
        ax[1,2].set_adjustable('box-forced')
        ax[1,2].set_xlabel('probmap')
        
        # masked prob map, use the same threshold as Hongbo's experiments = 0.615, with an average FP/normal = 30%
        mx_probmap = ma.masked_array(probmap, mask=probmap < 0.615)
        print "masked mx_probmap, mean detection prob = %f" % mx_probmap.mean()
        masked_probmap = ma.filled(mx_probmap, fill_value=0.0)
        
        # Display masked probmap when prob only >0.615
        ax[1,3].plot(coords_breastm[:, 1], coords_breastm[:, 0], '-r', linewidth=2)
        ax[1,3].axis((0, breastm[centroid[2],:,:].shape[1], breastm[centroid[2],:,:].shape[0], 0))
        ax[1,3].imshow(masked_probmap[centroid[2],:,:], cmap=plt.cm.gray)
        ax[1,3].set_adjustable('box-forced')
        ax[1,3].set_xlabel('probmap > 0.615')

        # Using the “marching squares” method to compute a the iso-valued contours 
        # from prob map
        outlines_probmap = find_contours(masked_probmap[centroid[2],:,:], 0)
        coords_probmap = []
        ax[1,4].plot(coords_breastm[:, 1], coords_breastm[:, 0], '-r', linewidth=2)
        ax[1,4].axis((0, breastm[centroid[2],:,:].shape[1], breastm[centroid[2],:,:].shape[0], 0))
        
        for oi, outline in enumerate(outlines_probmap):
            coords_probmap.append( approximate_polygon(outline, tolerance=2) )
            print "Number of points: %d to %d = %d percent reduction" % ( len(outline), len(coords_probmap[oi]), float(len(outline)-len(coords_probmap[oi]))/len(outline)*100 )
        
            ax[1,4].plot(coords_probmap[oi][:, 1], coords_probmap[oi][:, 0], '-g', linewidth=2)
            ax[1,4].axis((0, breastm[centroid[2],:,:].shape[1], breastm[centroid[2],:,:].shape[0], 0))
 
        ax[1,4].set_xlabel('outlines')
        
        #############################
        ###### 3) Use masked prob map only extract regions of the image with prob lesion > 0.615
        ## use the same threshold as Hongbo's experiments = 0.615, with an average FP/normal = 30%
        #############################        
        mask_mriVols = []
        for k in range(5):
            # implement using MaskArrays numpy.ma, where probability map is 50% or higher
            mx = ma.masked_array(mriVols[k], mask=probmap < 0.615)
            print "masked mriVol_%i, probmap > 0.615 mean SI/enhancement = %f" % (k, mx.mean())
            mask_mriVols.append( ma.filled(mx, fill_value=0.0) )
    
            ax[2,k].imshow(mask_mriVols[k][centroid[2],:,:], cmap=plt.cm.gray)
            ax[2,k].set_adjustable('box-forced')
            ax[2,k].set_xlabel(str(k)+'mask_'+sideB)
            
            # masked prob map
        mx_probmap = ma.masked_array(probmap, mask=probmap < 0.615)
        print "masked mx_probmap, mean detection prob = %f" % mx_probmap.mean()
        masked_probmap = ma.filled(mx_probmap, fill_value=0.0)

        #show and save                     
        fig.savefig( os.path.join( parseprobmaps_path,'{}_{}_{}_parse_imgprobmaps.pdf'.format(str(lesion_id),fStudyID.zfill(4),AccessionN) ) )
        
        ####################
        #### 4) generate delaunay triangulation of  a set of point locations
        ####################
        # start to find spread of lesion for query graphs (since its' in DICOM cords will be physical dim)
        sp_x = np.abs(np.abs(lesion_coords['segm_xmin'])-np.abs(lesion_coords['segm_xmax']))
        sp_y = np.abs(np.abs(lesion_coords['segm_ymin'])-np.abs(lesion_coords['segm_ymax']))
        sp_z = np.abs(np.abs(lesion_coords['segm_zmin'])-np.abs(lesion_coords['segm_zmax']))
        
        # phys coord = #pixels * pix_spacing
        probmap_sitk = sitk.ReadImage(probmap_filepath)
        vol_spacing = probmap_sitk.GetSpacing()
        
        pix_y = round(float(sp_y)/vol_spacing[0])
        pix_x = round(float(sp_x)/vol_spacing[1])
        
        # masked gt mx_side_gt
        mx_side_gt = ma.masked_array(side_gt_lesion, mask=side_gt_lesion==0)
        masked_side_gt = ma.filled(mx_side_gt, fill_value=0.0)

        outlines_side_gt = find_contours(masked_side_gt[centroid[2],:,:], 0)
        coords_side_gt = approximate_polygon( np.concatenate(outlines_side_gt), tolerance=2)
        
        
        mx_query = np.zeros(probmap.shape)
        ext_x = [int(ex) for ex in [np.min(coords_side_gt[:,1])-5,np.max(coords_side_gt[:,1])+5] ] # old way int(centroid[0]-pix_x/2),int(centroid[0]+pix_x/2)
        ext_y = [int(ey) for ey in [np.min(coords_side_gt[:,0])-5,np.max(coords_side_gt[:,0])+5] ] # int(centroid[1]-pix_y/2),int(centroid[1]+pix_y/2)
        mx_query[int(centroid[2]), ext_y[0]:ext_y[1], ext_x[0]:ext_x[1]] = 1
        
        mask_queryVols = []
        for k in range(5):
            mx = ma.masked_array(mriVols[k], mask=mx_query==0)
            print "masked lesionVol_%i, lesion mean SI/enhancement = %f" % (k, mx.mean())
            mask_queryVols.append( ma.filled(mx, fill_value=0.0) )
     
        mx_queryprob = ma.masked_array(probmap, mask=mx_query==0)
        mask_queryprob = ma.filled(mx_queryprob, fill_value=0.0)

        ####################
        # go through 2D slice centroid and extract contours from prob map
        ####################
        # find contours at the 2D slice centroid
        outlines_queryprob = find_contours(mask_queryprob[int(centroid[2]),:,:], 0.615)
        
        # once collected append them all as a 2D array of points
        allpts_queryprob = np.concatenate(outlines_queryprob)
        
        ####################
        # start with lesion triangulations
        ####################
        # perform Delaunay triangulation on the pts for query lesion
        a = np.asarray( [allpts_queryprob[:, 1], allpts_queryprob[:, 0]] ).transpose()
        lesion_delaunay = Delaunay(a)
        lesion_triangles = lesion_delaunay.points[lesion_delaunay.vertices]     
        
        # plot
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        ax = ax.flatten()
        ax[0].imshow(mask_queryVols[4][int(centroid[2]),:,:], cmap=plt.cm.gray)
        ax[0].axis((ext_x[0], ext_x[1], ext_y[1], ext_y[0]))
        ax[0].set_xlabel('4th postc Vol in centroid')
        ax[0].plot(lesion_delaunay.points[:, 0], lesion_delaunay.points[:, 1], '.r', linewidth=2)
        
        lines = []
        # Triangle vertices
        A = lesion_triangles[:, 0]
        B = lesion_triangles[:, 1]
        C = lesion_triangles[:, 2]
        lines.extend(zip(A, B))
        lines.extend(zip(B, C))
        lines.extend(zip(C, A))
        lines = LineCollection(lines, color='b')
        ax[1].imshow(mask_queryprob[int(centroid[2]),:,:], cmap=plt.cm.gray)
        ax[1].axis((ext_x[0], ext_x[1], ext_y[1], ext_y[0]))
        ax[1].plot(lesion_delaunay.points[:, 0], lesion_delaunay.points[:, 1], '.r', linewidth=2)
        ax[1].add_collection(lines)
        ax[1].set_xlabel('probmap Delaunay triangulated (query lesion)')     
        
        ####################
        # save lesion query graph
        # pickle file with delaunay.points, delaunay.vertices
        #   triangles = delaunay.points[delaunay.vertices] 
        # to read:
        #    with gzip.open(os.path.join(output_path,'{}_{}_{}_lesion_querygraph.pklz'.format(str(lesion_id),fStudyID.zfill(4),AccessionN)), 'rb') as f:
        #        try:
        #            lesion_triangles = pickle.load(f, encoding='latin1')
        #        except:
        #            lesion_triangles = pickle.load(f)
        ####################
        queryg = gzip.open(os.path.join(output_path,'{}_{}_{}_lesion_querygraph.pklz'.format(str(lesion_id),fStudyID.zfill(4),AccessionN)), 'wb')
        pickle.dump(lesion_delaunay, queryg, protocol=pickle.HIGHEST_PROTOCOL)
        queryg.close()
        #show and save                     
        fig.savefig( os.path.join( parseprobmaps_path,'{}_{}_{}_delaunay_query_lesion.pdf'.format(str(lesion_id),fStudyID.zfill(4),AccessionN) ) )
                
        ####################
        # Display non query points: go through the rest of 2D slices and extract contours from prob map
        #################### 
        mx_nonqueryprob = ma.masked_array(probmap, mask=mx_query==1)
        mask_nonqueryprob = ma.filled(mx_nonqueryprob, fill_value=0.0)
        # Display probmap, probmap outlines (except those of lesion region), breast outlines and combination 
        fig, ax = plt.subplots(nrows=int(probmap.shape[0]/round(np.sqrt(probmap.shape[0]))), ncols=int(np.sqrt(probmap.shape[0]))+2, figsize=(20, 10))
        ax = ax.flatten()
        
        for zslice in range(probmap.shape[0]):
            outlines_queryprob = find_contours(mask_nonqueryprob[zslice,:,:], 0.615)
            if(outlines_queryprob):
                outlines_pts = np.concatenate(outlines_queryprob)
                
                # add breast mask outline
                outline_breastm = find_contours(breastm[zslice,:,:], 0)[0]
                coords_breastm = approximate_polygon(outline_breastm, tolerance=2)

                ax[zslice].plot(coords_breastm[:, 1], coords_breastm[:, 0], '.b', linewidth=2)
                ax[zslice].axis((0, probmap.shape[1], probmap.shape[2], 0))

                ax[zslice].imshow(mask_nonqueryprob[zslice,:,:], cmap=plt.cm.gray)
                ax[zslice].set_adjustable('box-forced')
                ax[zslice].plot(outlines_pts[:, 1], outlines_pts[:, 0], '.r', linewidth=2)
                
                # combine and simplify all
                outlines_all = np.concatenate([outlines_pts, coords_breastm])      
                coords_all = approximate_polygon(outlines_all, tolerance=2)
                ax[zslice].plot(coords_all[:, 1], coords_all[:, 0], '.g', linewidth=2)
                # turn axes off
                ax[zslice].get_xaxis().set_visible(False)
                ax[zslice].get_yaxis().set_visible(False)
                
        #show and save                     
        fig.savefig( os.path.join( parseprobmaps_path,'{}_{}_{}_outlines_nonquery.pdf'.format(str(lesion_id),fStudyID.zfill(4),AccessionN) ) )
        
        ####################
        # Display non query triangulations: go through the rest of 2D slices and extract contours from prob map
        #################### 
        # Based on outlines Display triangulation of outlines combined
        fig, ax = plt.subplots(nrows=int(probmap.shape[0]/round(np.sqrt(probmap.shape[0]))), ncols=int(np.sqrt(probmap.shape[0]))+2, figsize=(20, 10))
        ax = ax.flatten()    
        # collect graphs of nonquery traingulations
        alls_nonqueryg_delaunay =[]        
        for zslice in range(probmap.shape[0]):
            outlines_queryprob = find_contours(mask_nonqueryprob[zslice,:,:], 0.615)
            if(outlines_queryprob):
                outlines_pts = np.concatenate(outlines_queryprob)
                
                # add breast mask outline
                outline_breastm = find_contours(breastm[zslice,:,:], 0)[0]
                coords_breastm = approximate_polygon(outline_breastm, tolerance=2)

                ax[zslice].plot(coords_breastm[:, 1], coords_breastm[:, 0], '.b', linewidth=2)
                ax[zslice].axis((0, probmap.shape[1], probmap.shape[2], 0))

                ax[zslice].imshow(mask_nonqueryprob[zslice,:,:], cmap=plt.cm.gray)
                ax[zslice].set_adjustable('box-forced')
                ax[zslice].plot(outlines_pts[:, 1], outlines_pts[:, 0], '.r', linewidth=2)
                
                # combine and simplify all
                outlines_all = np.concatenate([outlines_pts, coords_breastm])      
                coords_all = approximate_polygon(outlines_all, tolerance=2)
                
                ################
                # Add Delaunay triangulation
                ################
                # perform Delaunay triangulation on the pts for query lesion
                a = np.asarray( [coords_all[:, 1], coords_all[:, 0]] ).transpose()
                nonqueryg_delaunay = Delaunay(a)
                lesion_triangles = nonqueryg_delaunay.points[nonqueryg_delaunay.vertices]
                ax[zslice].plot(nonqueryg_delaunay.points[:, 0], nonqueryg_delaunay.points[:, 1], '.g', linewidth=2)
                  
                lines = []
                # Triangle vertices
                A = lesion_triangles[:, 0]
                B = lesion_triangles[:, 1]
                C = lesion_triangles[:, 2]
                lines.extend(zip(A, B))
                lines.extend(zip(B, C))
                lines.extend(zip(C, A))
                lines = LineCollection(lines, color='c')
                ax[zslice].add_collection(lines)
                # turn axes off
                ax[zslice].get_xaxis().set_visible(False)
                ax[zslice].get_yaxis().set_visible(False)
                
                # append and continue to next slice
                alls_nonqueryg_delaunay.append(nonqueryg_delaunay)                
        
        ####################
        # save all slices collection of query graph triangulations
        ####################
        nonqueryg  = gzip.open(os.path.join(output_path,'{}_{}_{}_non_querygraphs_Delatriang.pklz'.format(str(lesion_id),fStudyID.zfill(4),AccessionN)), 'wb')
        pickle.dump(alls_nonqueryg_delaunay, nonqueryg, protocol=pickle.HIGHEST_PROTOCOL)
        nonqueryg.close()
        
        #show and save
        fig.savefig( os.path.join( parseprobmaps_path,'{}_{}_{}_delaunay_nonquery.pdf'.format(str(lesion_id),fStudyID.zfill(4),AccessionN) ))
        
        ####################
        # not Display but save all queries (lesion + non query triangulations combined)
        #################### 
        alls_queryg_delaunay =[]        
        for zslice in range(probmap.shape[0]):
            outlines_queryprob = find_contours(mask_nonqueryprob[zslice,:,:], 0.615)
            if(outlines_queryprob):
                outlines_pts = np.concatenate(outlines_queryprob)
                
                # add breast mask outline
                outline_breastm = find_contours(breastm[zslice,:,:], 0)[0]
                coords_breastm = approximate_polygon(outline_breastm, tolerance=2)

                # combine and simplify all
                outlines_all = np.concatenate([outlines_pts, coords_breastm])      
                coords_all = approximate_polygon(outlines_all, tolerance=2)
                
                ################
                # Add Delaunay triangulation
                ################
                # perform Delaunay triangulation on the pts for query lesion
                a = np.asarray( [coords_all[:, 1], coords_all[:, 0]] ).transpose()
                allqueryg_delaunay = Delaunay(a)                
                # append and continue to next slice
                alls_queryg_delaunay.append(allqueryg_delaunay)                
        
        ####################
        # save all slices collection of query graph triangulations
        ####################
        allqueryg   = gzip.open(os.path.join(output_path,'{}_{}_{}_allquerygraphs_Delatriang.pklz'.format(str(lesion_id),fStudyID.zfill(4),AccessionN)), 'wb')
        pickle.dump(alls_queryg_delaunay, allqueryg, protocol=pickle.HIGHEST_PROTOCOL)
        allqueryg.close()
        
        #show and save
        fig.savefig( os.path.join( parseprobmaps_path,'{}_{}_{}_delaunay_nonquery.pdf'.format(str(lesion_id),fStudyID.zfill(4),AccessionN) ))
        
        
        # explore resulting graph in interesting sites  
        print "======================================\n Centroid Slice: %i \n======================================\n" % int(centroid[2])
        fig, ax = plt.subplots(nrows=1,ncols=3, figsize=(20, 10))
        zslice = int(centroid[2])
        ploi = 0  
        plotSlice_outlines(mask_nonqueryprob, breastm, zslice, ax, ploi) 
        ploi = 1
        plotSlice_outlines(probmap, breastm, zslice, ax, ploi)
        ploi = 2
        plotSlice_outlines(probmap, breastm, zslice, ax, ploi, bg_img=mriVols[4])
        
        #show and save  
        plt.show(block=False)                    
        fig.savefig( os.path.join( parseprobmaps_path,'{}_{}_{}_zoomin_w4thpost_{}slice_{}.pdf'.format(str(lesion_id),fStudyID.zfill(4),AccessionN,str(zslice),str(cond+BenignNMaligNAnt)) ))
        plt.close("all")
        
        ## next line
        line = file_ids.readline()       

    return
    

if __name__ == '__main__':
    run_parseprobmaps()    
    
    