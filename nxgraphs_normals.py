# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 12:11:42 2016

@author: DeepLearning
"""
import pandas as pd
import os
import os.path, sys
import shutil
import glob
import numpy as np
import random
import SimpleITK as sitk

# to query local databse
from mylocalbase import localengine
from sqlalchemy.orm import sessionmaker
import mylocaldatabase_new

import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
sns.set(color_codes=True)

import numpy.ma as ma
from skimage.measure import find_contours, approximate_polygon

from scipy.spatial import Delaunay
from matplotlib.collections import LineCollection

# to save graphs
import six.moves.cPickle as pickle
import gzip

import networkx as nx


def run_nxGraphs_normals():
    # Get Root folder ( the directory of the script being run)
    path_rootFolder = os.path.dirname(os.path.abspath(__file__))
    os.chdir(path_rootFolder) 
    
    processed_path = 'Y:\\Hongbo\\processed_data'
    probmaps_path = 'Y:\\Hongbo\\segmentations_train'
    graphs_path = 'Z:\\Cristina\\Section3\\breast_MR_pipeline\\graphs'
    triangulation_path = 'Z:\\Cristina\\Section3\\breast_MR_pipeline\\triangulations'

    if not os.path.exists(triangulation_path):
        os.mkdir(triangulation_path)
        
    file_ids = open("listofprobmaps_normals.txt","r") 
    file_ids.seek(0)
    line = file_ids.readline()
  
    inormal=0
    while ( line ) : 
        # Get the line: id  CAD study #	Accession #	Date of Exam	Pathology Diagnosis 
        print(line)
        fileline = line.split('_')
        lesion_id = 'n'+str(inormal) 
        oldlesion_id = int(fileline[0])        
        fStudyID = fileline[1] 
        AccessionN = fileline[2]  
        sideB = fileline[3]
    
        #############################
        ###### 1) Accesing mc images, prob maps, gt_lesions and breast masks
        #############################
        # get dynmic series info
        glob_result = glob.glob( os.path.join(processed_path,'{}_{}_{}_'.format(str(oldlesion_id),fStudyID.zfill(4),AccessionN)+'*@*'))
        onlydyn_results = [os.path.basename(s) for s in glob_result  if "@" in s]        
        globdyn_series =  [s.split('@')[0].split('_') for s in onlydyn_results  if "@" in s]
        precontrast_id = np.min([int(s[-1]) for s in globdyn_series])
        DynSeries_nums = [str(n) for n in range(precontrast_id,precontrast_id+5)]
        
        DynSeries_imagefiles = []
        mriVols = []
        print "Reading MRI volumes..."
        for j in range(5):
            #the output mha:lesionid_patientid_access#_series#@acqusionTime.mha
            DynSeries_filename = '{}_{}_{}_{}'.format(str(oldlesion_id),fStudyID.zfill(4),AccessionN,DynSeries_nums[j] )
            
            #write log if mha file not exist             
            glob_result = glob.glob(os.path.join(processed_path,DynSeries_filename+'*')) #'*':do not to know the exactly acquistion time
            if glob_result != []:
                filename = glob_result[0]
                
            # add side info from the side of the lesion
            filename_wside = filename[:-4]+'_{}.mha'.format(sideB)
            DynSeries_imagefiles.append(filename_wside)
               
            # read Volumnes
            mriVols.append( sitk.GetArrayFromImage(sitk.Cast(sitk.ReadImage(DynSeries_imagefiles[j]),sitk.sitkFloat32)) )
        
        #############################
        ###### 3) load graph object into memory
        ## eg. n0_0001_7575429_left_allquerygraphs_Delatriang
        ############################# 
#        with gzip.open( os.path.join(graphs_path,'{}_{}_{}_{}_allquerygraphs_Delatriang.pklz'.format(str(lesion_id),fStudyID.zfill(4),AccessionN,sideB)), 'rb') as f:
#            normals_delaunay = pickle.load(f)
        
        # read only one bilateral volume for left/right splitting
        bilateral_filename = '{}_{}_{}_{}@'.format(str(oldlesion_id),fStudyID.zfill(4),AccessionN,DynSeries_nums[1] )
        bilateral_filepath = glob.glob(os.path.join( processed_path,bilateral_filename+'*'+'_mc.mha' ))[0]
        bilateral_Vol = sitk.GetArrayFromImage(sitk.Cast(sitk.ReadImage(bilateral_filepath),sitk.sitkFloat32)) 
        # get original No of slices
        nslices = bilateral_Vol.shape[0]
        lefts = int(round(nslices/2))
        
        # select a random slice number
        rands = random.randrange(5,lefts-5)
        
        #get wofs filename
        print "Reading probability map..."
        probmap_filename = '{}_{}_{}_{}#2_1_NS_NF_probmap.mha'.format(str(oldlesion_id),fStudyID.zfill(4),AccessionN,sideB)
        probmap_filepath = os.path.join(probmaps_path,probmap_filename)
        probmap = sitk.GetArrayFromImage(sitk.Cast( sitk.ReadImage(probmap_filepath), sitk.sitkFloat32)) 
        
        print "Reading breast mask..." 
        breastm_filename = '{}_{}_{}_wofs_reg_breastmask_{}.mha'.format(str(oldlesion_id),fStudyID.zfill(4),AccessionN,sideB)
        breastm_filepath = os.path.join(processed_path,breastm_filename)
        breastm = sitk.GetArrayFromImage(sitk.Cast( sitk.ReadImage(breastm_filepath), sitk.sitkFloat32))
 
        ####################
        #### 4) And compute SER Volume
        ####################       
        mask_mriVols = []
        onlyROI = []
        for k in range(5):
            mx = ma.masked_array(mriVols[k], mask=breastm==0)
            print "masked breastVol_%i, breast mean SI/enhancement = %f" % (k, mx.mean())
            mask_mriVols.append( ma.filled(mx, fill_value=None) )
            onlyROI.append( ma.compress_rowcols( mask_mriVols[k][rands,:,:] ))
        
        # Compute ser volume (SERvol1/SERvol4)
        SERvol1 = (onlyROI[1] - onlyROI[0])
        SERvol4 = (onlyROI[4] - onlyROI[0])
        # accoumt for zero value pixels in the denominator
        SERvol4nonz = np.asarray([pix if pix != 0.0 else 0.01 for pix in SERvol4.flatten()]).reshape(SERvol4.shape) 
        # compute SER, turned off pixels will have spurious high values
        SER = np.asarray(SERvol1/SERvol4nonz)
        # clip extremely low and high values ( see ipython notebook for SERexploration)
        # explore other values for SER clipping
        nonzSER =  [pix if pix >= 0.0 else -0.01 for pix in SER.flatten()]
        nonzmax1p1SER =  np.asarray([pix if pix <= 2 else 2.5 for pix in nonzSER]).reshape(SER.shape)
        nonzmax1p1SER = nonzmax1p1SER.astype(np.float32)
        
        # Display 
        fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(20, 12))
        for k in range(5):
            ax[0,k].imshow(mriVols[k][rands,:,:], cmap=plt.cm.gray)
            ax[0,k].set_adjustable('box-forced')
            ax[0,k].set_xlabel(str(k)+'mc_'+sideB)
            
        ax[1,0].imshow(SERvol1, cmap=plt.cm.gray)
        ax[1,0].set_adjustable('box-forced')
        ax[1,0].set_xlabel('SERvol1')
        
        ax[1,1].imshow(SERvol4, cmap=plt.cm.gray)
        ax[1,1].set_adjustable('box-forced')
        ax[1,1].set_xlabel('SERvol4')
        
        ax[1,2].imshow(nonzmax1p1SER, cmap=plt.cm.gray)
        ax[1,2].set_adjustable('box-forced')
        ax[1,2].set_xlabel(str(np.histogram(nonzmax1p1SER)))
        
        # SER Histogram
        pixSER = [pix for pix in nonzmax1p1SER.ravel() if pix !=0]
        N, bins, patches = ax[1,3].hist(pixSER, 20)
        # we need to normalize the data to 0..1 
        fracs = N.astype(float)/N.max()
        norm = matplotlib.colors.Normalize(fracs.min(), fracs.max())
        
        for thisfrac, thispatch in zip(fracs, patches):
            color = plt.cm.cool(norm(thisfrac))
            thispatch.set_facecolor(color) 
        ax[1,3].set_xlabel('SER Histogram in lesion')

        #show and save  
        fig.savefig( os.path.join( triangulation_path,'{}_{}_{}_{}_mriVols_{}Wholeslice_normal.pdf'.format(str(lesion_id),fStudyID.zfill(4),AccessionN,sideB,str(rands)) ))

        ##############
        ## 5) Add normals delaunay triangulation
        ##############
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 12))
        ax = axes.flatten()
        # Overlay the three images
        im = ax[0].imshow(nonzmax1p1SER, cmap=plt.cm.cool, interpolation='none', clim=[min(nonzmax1p1SER.ravel()), max(nonzmax1p1SER.ravel())])
        ax[0].set_xlabel('Dela. Triangulation + SER')
        
        # Set the colormap and norm to correspond to the data for which
        divider = make_axes_locatable(ax[0])
        caxSER = divider.append_axes("right", size="20%", pad=0.05)
        plt.colorbar(im, cax=caxSER)
        
        ## Used only normal prob map 
        # masked prob map, use the same threshold as Hongbo's experiments = 0.615, with an average FP/normal = 30%
        mx_probmap = ma.masked_array(probmap, mask=probmap < 0.615)
        print "masked mx_probmap, mean detection prob = %f" % mx_probmap.mean()
        masked_probmap = ma.filled(mx_probmap, fill_value=0.0)
        
        # Using the “marching squares” method to compute a the iso-valued contours 
        # from prob map
        outlines_probmap = find_contours(masked_probmap[rands,:,:], 0)
        normals_delaunays_all = []
        for oi, outline in enumerate(outlines_probmap):
            coords_probmap = approximate_polygon(outline, tolerance=0.5)
            
            if(len(coords_probmap)>6): # min 6 n points or more, for min 4 in the hull, gives 6 triangles
                # Theorem 9.1 Let P be a set of n points in the plane, not all collinear, 
                # and let k denote the number of points in P that lie on the boundary of the convex hull of P. 
                # Then any triangulation of P has 2n−2−k triangles and 3n−3−k edges.
                # FOR A minimum of 6 triangles: 6 = 2n-2-k, if k=4 then n=6,(min # of points in the hull or coords_probmap)
                print "Number of points: %d to %d = %d percent reduction" % ( len(outline), len(coords_probmap), float(len(outline)-len(coords_probmap))/len(outline)*100 )
         
                ################
                # Add Delaunay triangulation
                ################
                # perform Delaunay triangulation on the pts for query lesion
                a = np.asarray( [coords_probmap[:, 1], coords_probmap[:, 0]] ).transpose()
                normals_delaunay = Delaunay(a)
                normals_triangles = normals_delaunay.points[normals_delaunay.vertices]   
                ## Plot delaunay triangulation
                ax[1].imshow(probmap[rands,:,:], cmap=plt.cm.gray)
                ax[1].plot(normals_delaunay.points[:, 0], normals_delaunay.points[:, 1], '.g', linewidth=2)
                lines = []
                # Triangle vertices
                A = normals_triangles[:, 0]
                B = normals_triangles[:, 1]
                C = normals_triangles[:, 2]
                lines.extend(zip(A, B))
                lines.extend(zip(B, C))
                lines.extend(zip(C, A))
                lines = LineCollection(lines, color='b')
                ax[1].add_collection(lines)
                # append normal patterns
                normals_delaunays_all.append(normals_delaunay)
                
                ## Plot mriVol correlation
                ax[2].imshow(mriVols[4][rands,:,:], cmap=plt.cm.gray)
                ax[2].plot(normals_delaunay.points[:, 0], normals_delaunay.points[:, 1], '.r', linewidth=2)
                
        if(len(normals_delaunays_all)>0):
            # compute detection soleley based on thresholded probmap        
            print "%d FP/slice: will reduce with normal graph pattern mining (hypothesis)" % len(normals_delaunays_all)
            ax[1].axis((0, probmap[rands,:,:].shape[1], probmap[rands,:,:].shape[0], 0))
            ax[2].axis((0, probmap[rands,:,:].shape[1], probmap[rands,:,:].shape[0], 0))
            ax[1].set_xlabel('Probmap + all Normal Dela. Triangulations/slice = '+ str(len(normals_delaunays_all))+' FP/slice')
            ax[2].set_xlabel('mriVol postc4 + all Normal detections/slice = '+ str(len(normals_delaunays_all))+' FP/slice')
    
            # adjust visualization axes to largest detection
            listnpts = [len(npts.points) for npts in normals_delaunays_all]
            largest_normal_delaunay = normals_delaunays_all[ listnpts.index(np.max(listnpts)) ]
            largest_normal_triangles = largest_normal_delaunay.points[largest_normal_delaunay.vertices]   
            minTriag = largest_normal_delaunay.points.min(axis=0)
            maxTriag = largest_normal_delaunay.points.max(axis=0)
            
            ax[3].imshow(nonzmax1p1SER, cmap=plt.cm.cool, interpolation='none', clim=[min(nonzmax1p1SER.ravel()), max(nonzmax1p1SER.ravel())])
            ax[3].plot(largest_normal_delaunay.points[:, 0], largest_normal_delaunay.points[:, 1], '.g', linewidth=2)
            ax[3].axis((minTriag[0]-5, maxTriag[0]+5, maxTriag[1]+5, minTriag[1]-5))
            lines = []
            # Triangle vertices
            A = largest_normal_triangles[:, 0]
            B = largest_normal_triangles[:, 1]
            C = largest_normal_triangles[:, 2]
            lines.extend(zip(A, B))
            lines.extend(zip(B, C))
            lines.extend(zip(C, A))
            lines = LineCollection(lines, color='b')
            ax[3].add_collection(lines)
            ax[3].set_xlabel('largest Normal Dela. Triangulation + SER')
    
            #########################
            # 6) Calculate triangulation faces, compute faces centroid, use as new triangulation points
            #########################
            trianF = []
            for verT in largest_normal_triangles:
                trianF.append( list(verT.mean(axis=0) ))
                
            a = np.asarray( [np.asarray(trianF)[:,0], np.asarray(trianF)[:,1] ]).transpose()
            trianglFaces_delaunay = Delaunay(a)
            
            #############################
            ###  7) Create triangulation as a nx graph, Add weights inversely proportional to SER, 
            ###  if SER is high (greater than 1.1 the signal intensity decreases at post4 --> washout) --> weight =1/SER is low
            ###  if SER is medium (between 0.9 and 1.1 represents a plateau at post4) --> weight =1/SER is medium
            ###  if SER is low ( less than 0.9 indicates that signal intensity continues to rise at post4) --> weight =1/SER is High
            ###  minimum spanning tree M will add only links to low weight edges, to minimize total graph weigth 
            #############################       
            # The graph G can be grown in several ways. NetworkX includes many graph generator functions and facilities 
            # get list of delaunay points        
            pts = [tuple(pi.flatten()) for pi in trianglFaces_delaunay.points]            
    
            #############################
            ###### 4) Sample SER at nodes
            #############################
            nodew = []
            print "nodeweights" 
            for node in pts:
                SERloc = tuple([int(loc) for loc in node])
                #print nonzmax1p1SER[SERloc[1],SERloc[0]]
                nodew.append( nonzmax1p1SER[SERloc[1],SERloc[0]] )
     
            # create placeholder for nx nodes
            nodes = list(range(len(pts)))
            # mapping from vertices to nodes
            m = dict(enumerate(nodes)) 
            
            #Create a graph
            anormalG = nx.Graph() 
            print "Average triangle weight"         
            for i in range(trianglFaces_delaunay.nsimplex):
                wp1 = np.max([nodew[m[trianglFaces_delaunay.vertices[i,0]]], nodew[m[trianglFaces_delaunay.vertices[i,1]]]])
                anormalG.add_edge( m[trianglFaces_delaunay.vertices[i,0]], m[trianglFaces_delaunay.vertices[i,1]], weight = wp1 )
                wp2 = np.max([nodew[m[trianglFaces_delaunay.vertices[i,1]]], nodew[m[trianglFaces_delaunay.vertices[i,2]]]])
                anormalG.add_edge( m[trianglFaces_delaunay.vertices[i,1]], m[trianglFaces_delaunay.vertices[i,2]], weight = wp2 )
                wp3 = np.max([nodew[m[trianglFaces_delaunay.vertices[i,2]]], nodew[m[trianglFaces_delaunay.vertices[i,0]]]])
                anormalG.add_edge( m[trianglFaces_delaunay.vertices[i,2]], m[trianglFaces_delaunay.vertices[i,0]], weight = wp3 )
                print(np.mean([wp1, wp2, wp3])) 
                                 
            # get original position of points
            pos = dict(zip(nodes,pts))
            # draw
            ax[4].imshow(mriVols[4][rands,:,:], cmap=plt.cm.gray)
            ax[4].axis((minTriag[0]-5, maxTriag[0]+5, maxTriag[1]+5, minTriag[1]-5))
            nx.draw_networkx(anormalG, pos, ax=ax[4], with_labels=False, node_color='c',node_size=10, edge_color='b', width=1.5)
            ax[4].set_xlabel('4th postc mriVol + FacesTriangulated normal detection graph')  
            
            #############################
            ###### 5) Calculate minimum spanning Tree (MST)
            #############################
            MST = nx.minimum_spanning_edges( anormalG, data=True)
            edgelist = list(MST)
            
            #Convert minimum spanning tree into graph to be able to depth searched
            MST_lesionG = nx.from_edgelist(edgelist)
            print "all edges"
            print MST_lesionG.edges(data=True)
            SER_nodeweights = [d['weight'] for (u,v,d) in MST_lesionG.edges(data=True)]
            
            # draw gaprh
            nxg = nx.draw_networkx_edges(MST_lesionG, pos, ax=ax[4], edge_color=SER_nodeweights, edge_cmap=plt.cm.inferno, 
                                         edge_vmin=min(SER_nodeweights),edge_vmax=max(SER_nodeweights), width=1.5)
            v = np.linspace(min(SER_nodeweights), max(SER_nodeweights), 10, endpoint=True)
            
            divider = make_axes_locatable(ax[4])
            caxEdges = divider.append_axes("right", size="20%", pad=0.05)
            plt.colorbar(nxg, cax=caxEdges, ticks=v)
    
            # Overlay with the mri image
            nxg = nx.draw_networkx_edges(MST_lesionG, pos, ax=ax[5], edge_color=SER_nodeweights, edge_cmap=plt.cm.inferno, 
                                         edge_vmin=min(SER_nodeweights),edge_vmax=max(SER_nodeweights), width=1.5)
            nx.draw_networkx_nodes(MST_lesionG, pos, ax=ax[5], node_size=25, node_shape='.', node_color='g', with_labels=False)
            ax[5].imshow(mriVols[4][rands,:,:], cmap=plt.cm.gray)
            ax[5].axis((minTriag[0]-5, maxTriag[0]+5, maxTriag[1]+5, minTriag[1]-5))
            ax[5].set_xlabel('MST FacesTriangulated normal detection graph + SER edge weighted')
            
            #show and save  
            plt.show(block=False)                    
            fig.savefig( os.path.join( triangulation_path,'{}_{}_{}_{}_DelauTriangFaces_{}zoomInSlice_normal.pdf'.format(str(lesion_id),fStudyID.zfill(4),AccessionN,sideB,str(rands)) ))
        else:
            print "No FP/slice"

        #############################
        ###### COMPLETE FOR ALL NORMAL SLICES
        #############################      
        allslices_anormalG = []
        allslices_MST_anormalG = []
        
        mask_mriVols = []
        for k in range(5):
            mx = ma.masked_array(mriVols[k], mask=breastm==0)
            print "masked breastVol_%i, breast mean SI/enhancement = %f" % (k, mx.mean())
            mask_mriVols.append( ma.filled(mx, fill_value=None) )
        
        # Compute ser volume (SERvol1/SERvol4) now is 3D volumes
        SERvol1 = (mask_mriVols[1] - mask_mriVols[0])
        SERvol4 = (mask_mriVols[4] - mask_mriVols[0])
        # accoumt for zero value pixels in the denominator
        SERvol4nonz = np.asarray([pix if pix != 0.0 else 0.01 for pix in SERvol4.flatten()]).reshape(SERvol4.shape) 
        # compute SER, turned off pixels will have spurious high values
        SER = np.asarray(SERvol1/SERvol4nonz)
        # clip extremely low and high values ( see ipython notebook for SERexploration)
        # explore other values for SER clipping
        nonzSER =  [pix if pix >= 0.0 else -0.01 for pix in SER.flatten()]
        nonzmax1p1SER =  np.asarray([pix if pix <= 2 else 2.5 for pix in nonzSER]).reshape(SER.shape)
        nonzmax1p1SER = nonzmax1p1SER.astype(float32)
        
        ## Used only normal prob map 
        # masked prob map, use the same threshold as Hongbo's experiments = 0.615, with an average FP/normal = 30%
        mx_probmap = ma.masked_array(probmap, mask=probmap < 0.615)
        print "masked mx_probmap, mean detection prob = %f" % mx_probmap.mean()
        masked_probmap = ma.filled(mx_probmap, fill_value=0.0)
        
        ####################
        ### 6) Iterate through each slice and represent each slice as a nx graph
        ####################   
        nslices = breastm.shape[0]
        # Display 
        figSER, axesSER = plt.subplots(nrows=int(np.sqrt(nslices)), ncols=int(np.sqrt(nslices))+2, figsize=(20, 12))
        axSER = axesSER.flatten()
        figMST, axesMST = plt.subplots(nrows=int(np.sqrt(nslices)), ncols=int(np.sqrt(nslices))+2, figsize=(20, 12))
        axMST = axesMST.flatten()
         
        for kslice in range(nslices):
            print "\n============ slice %d" %  kslice
            # Overlay the SER
            im = axSER[kslice].imshow(nonzmax1p1SER[kslice,:,:], cmap=plt.cm.cool, interpolation='none', 
                    clim=[min(nonzmax1p1SER[kslice,:,:].ravel()), max(nonzmax1p1SER[kslice,:,:].ravel())])
            axSER[kslice].set_xlabel('Dela. Triangulation + SER')
    
            # Set the colormap and norm to correspond to the data for which
            divider = make_axes_locatable(axSER[kslice])
            caxSER = divider.append_axes("right", size="20%", pad=0.05)
            plt.colorbar(im, cax=caxSER)
        
            # Using the “marching squares” method to compute a the iso-valued contours 
            # from prob map
            outlines_probmap = find_contours(masked_probmap[kslice,:,:], 0)
            # initialize contents
            normals_delaunays_all = []
            aslice_anormalG = []
            aslice_MST_anormalG = []
            for oi, outline in enumerate(outlines_probmap):
                coords_probmap = approximate_polygon(outline, tolerance=0.5)
                
                if(len(coords_probmap)>6): # min 6 n points or more, for min 4 in the hull, gives 6 triangles
                # Theorem 9.1 Let P be a set of n points in the plane, not all collinear, 
                # and let k denote the number of points in P that lie on the boundary of the convex hull of P. 
                # Then any triangulation of P has 2n−2−k triangles and 3n−3−k edges.
                # FOR A minimum of 6 triangles: 6 = 2n-2-k, if k=4 then n=6,(min # of points in the hull or coords_probmap)      
                    print "Number of points: %d to %d = %d percent reduction" % ( len(outline), len(coords_probmap), float(len(outline)-len(coords_probmap))/len(outline)*100 )         
                    ################
                    # Add Delaunay triangulation
                    ################
                    # perform Delaunay triangulation on the pts for query lesion
                    a = np.asarray( [coords_probmap[:, 1], coords_probmap[:, 0]] ).transpose()
                    normals_delaunay = Delaunay(a)
                    normals_triangles = normals_delaunay.points[normals_delaunay.vertices]   

                    # append normal patterns
                    normals_delaunays_all.append(normals_delaunay)
            
            # compute detection soleley based on thresholded probmap        
            print "%d FP/slice: will reduce with normal graph pattern mining (hypothesis)" % len(normals_delaunays_all)

            if(normals_delaunays_all):
                # iterate throught individual thrsholded detections/triangulations/plot and nx graph MST
                for kTriang in range(len(normals_delaunays_all)):            
                    anormal_delaunay = normals_delaunays_all[kTriang]
                    anormal_triangles = anormal_delaunay.points[anormal_delaunay.vertices]   
                    
                    ### filter only a minimum of 3 triangles
                    #if(len(anormal_triangles)>3):
                    #########################
                    # 6) Calculate triangulation faces, compute faces centroid, use as new triangulation points
                    #########################
                    trianF = []
                    for verT in anormal_triangles:
                        trianF.append( list(verT.mean(axis=0) ))
                        
                    a = np.asarray( [np.asarray(trianF)[:,0], np.asarray(trianF)[:,1] ]).transpose()
                    trianglFaces_delaunay = Delaunay(a)
                
                    #############################
                    ###  7) Create triangulation as a nx graph, Add weights inversely proportional to SER,  
                    # The graph G can be grown in several ways. NetworkX includes many graph generator functions and facilities 
                    # get list of delaunay points        
                    pts = [tuple(pi.flatten()) for pi in trianglFaces_delaunay.points]            
        
                    #############################
                    ###### 4) Sample SER at nodes
                    nodew = []
                    #print "nodeweights" 
                    for node in pts:
                        SERloc = tuple([int(loc) for loc in node])
                        #print nonzmax1p1SER[SERloc[1],SERloc[0]]
                        nodew.append( nonzmax1p1SER[kslice,SERloc[1],SERloc[0]] )
         
                    # create placeholder for nx nodes
                    #print(nodew)
                    nodes = list(range(len(pts)))
                    # mapping from vertices to nodes
                    m = dict(enumerate(nodes)) 
                    # get original position of points
                    pos = dict(zip(nodes,pts))
                    
                    #Create a graph
                    anormalG = nx.Graph() 
                    for i in range(len(pts)):
                        # add position as node attributes
                        anormalG.add_node(i, pos=pos[i])
            
                    #print "Average triangle weight"         
                    for i in range(trianglFaces_delaunay.nsimplex):
                        # add edges with weigth attributes
                        wp1 = np.max([nodew[m[trianglFaces_delaunay.vertices[i,0]]], nodew[m[trianglFaces_delaunay.vertices[i,1]]]])
                        anormalG.add_edge( m[trianglFaces_delaunay.vertices[i,0]], m[trianglFaces_delaunay.vertices[i,1]], weight = wp1 )
                        wp2 = np.max([nodew[m[trianglFaces_delaunay.vertices[i,1]]], nodew[m[trianglFaces_delaunay.vertices[i,2]]]])
                        anormalG.add_edge( m[trianglFaces_delaunay.vertices[i,1]], m[trianglFaces_delaunay.vertices[i,2]], weight = wp2 )
                        wp3 = np.max([nodew[m[trianglFaces_delaunay.vertices[i,2]]], nodew[m[trianglFaces_delaunay.vertices[i,0]]]])
                        anormalG.add_edge( m[trianglFaces_delaunay.vertices[i,2]], m[trianglFaces_delaunay.vertices[i,0]], weight = wp3 )
                        #print(np.mean([wp1, wp2, wp3])) 
                                                
                    ###### 5) Calculate minimum spanning Tree (MST)
                    #############################
                    MST = nx.minimum_spanning_edges( anormalG, data=True)
                    edgelist = list(MST)
                    
                    #Convert minimum spanning tree into graph to be able to depth searched
                    MST_lesionG = nx.from_edgelist(edgelist)
                    for i in range(len(pts)):
                        # add position as node attributes
                        MST_lesionG.node[i]['pos']=pos[i]
                        
                    #print "all edges"
                    #print MST_lesionG.edges(data=True)
                    SER_nodeweights = [d['weight'] for (u,v,d) in MST_lesionG.edges(data=True)]
                    
                    #############################
                    # draw graph
                    nx.draw_networkx(anormalG, pos, ax=axSER[kslice], with_labels=False, node_color='c',node_size=10, edge_color='b', width=1.5)

                    # Overlay with the mri image
                    nxg = nx.draw_networkx_edges(MST_lesionG, pos, ax=axMST[kslice], edge_color=SER_nodeweights, edge_cmap=plt.cm.inferno, 
                                                 edge_vmin=min(SER_nodeweights),edge_vmax=max(SER_nodeweights), width=1.5)
                    nx.draw_networkx_nodes(MST_lesionG, pos, ax=axMST[kslice], node_size=20, node_shape='.', node_color='g', with_labels=False)
            
                    # append to list per slice
                    aslice_anormalG.append( anormalG )
                    aslice_MST_anormalG.append( MST_lesionG )
                        
                # compute detection soleley based on thresholded probmap        
                print "%d FP/slice: reduced by inforcing min # of 3 triangles/detection graph" % len(aslice_anormalG)
                
                if(len(aslice_anormalG)>0):
                    ## Draw rest            
                    axMST[kslice].imshow(mriVols[4][kslice,:,:], cmap=plt.cm.gray)
                    # add colormap
                    v = np.linspace(-0.01, 2.5, 10, endpoint=True)
                    divider = make_axes_locatable(axMST[kslice])
                    caxEdges = divider.append_axes("right", size="20%", pad=0.05)
                    plt.colorbar(nxg, cax=caxEdges, ticks=v)
                    
                    # adjust visualization axes to largest detection
                    listnpts = [anormalG.number_of_nodes() for anormalG in aslice_anormalG]
                    largest_normal_delaunay = aslice_anormalG[ listnpts.index(np.max(listnpts)) ]
                    
                    # to restablish node positions
                    pts = [attr['pos'] for (n,attr) in largest_normal_delaunay.nodes(data=True)]
                    minTriag = np.asarray(pts).min(axis=0)
                    maxTriag = np.asarray(pts).max(axis=0)
                    # extend +/- 10 pixels
                    axSER[kslice].axis((minTriag[0]-5, maxTriag[0]+5, maxTriag[1]+5, minTriag[1]-5))
                    axMST[kslice].axis((minTriag[0]-5, maxTriag[0]+5, maxTriag[1]+5, minTriag[1]-5))
                    # turn axes off
                    axSER[kslice].get_xaxis().set_visible(False)
                    axSER[kslice].get_yaxis().set_visible(False)
                    axMST[kslice].get_xaxis().set_visible(False)
                    axMST[kslice].get_yaxis().set_visible(False)
                    
                    ## append to list of all slices
                    allslices_anormalG.append( aslice_anormalG )
                    allslices_MST_anormalG.append( aslice_MST_anormalG )
                else:
                    allslices_anormalG.append( [] )
                    allslices_MST_anormalG.append( [] )            
            else:
                allslices_anormalG.append( [] )
                allslices_MST_anormalG.append( [] )                

        #show and save  
        plt.show(block=False)                    
        figSER.savefig( os.path.join( triangulation_path,'{}_{}_{}_{}_DelauTriangFaces_SERallSlices_normal.pdf'.format(str(lesion_id),fStudyID.zfill(4),AccessionN,sideB) ))
        figMST.savefig( os.path.join( triangulation_path,'{}_{}_{}_{}_DelauTriangFaces_MSTallSlices_normal.pdf'.format(str(lesion_id),fStudyID.zfill(4),AccessionN,sideB) ))
        plt.close("all")

        ####################
        # save all slices collection of query graph triangulations
        ####################
        nxgraph_anormalG   = gzip.open(os.path.join(graphs_path,'{}_{}_{}_{}_FacesTriang_allSlices_nxgraph.pklz'.format(str(lesion_id),fStudyID.zfill(4),AccessionN,sideB)), 'wb')
        pickle.dump(allslices_anormalG, nxgraph_anormalG, protocol=pickle.HIGHEST_PROTOCOL)
        nxgraph_anormalG.close()
        
        nxgraph_MST_anormalG  = gzip.open(os.path.join(graphs_path,'{}_{}_{}_{}_MST_allSlices_nxgraph.pklz'.format(str(lesion_id),fStudyID.zfill(4),AccessionN,sideB)), 'wb')
        pickle.dump(allslices_MST_anormalG, nxgraph_MST_anormalG, protocol=pickle.HIGHEST_PROTOCOL)
        nxgraph_MST_anormalG.close()

        # TO LOAD
#        with gzip.open(os.path.join(graphs_path,'{}_{}_{}_{}_FacesTriang_allSlices_nxgraph.pklz'.format(str(lesion_id),fStudyID.zfill(4),AccessionN,sideB)), 'rb') as fu:
#            allSlices_nxgraph = pickle.load(fu)

        ## next line
        line = file_ids.readline()    
        inormal+=1
        
    return
    

if __name__ == '__main__':
    run_nxGraphs_normals()    
    
    