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
from parse_probmaps import querylocalDatabase_wRad

def run_nxGraphs():
    # Get Root folder ( the directory of the script being run)
    path_rootFolder = os.path.dirname(os.path.abspath(__file__))
    os.chdir(path_rootFolder) 
    
    processed_path = 'Y:\\Hongbo\\processed_data'
    probmaps_path = 'Y:\\Hongbo\\segmentations_train'
    gt_path = 'Y:\\Hongbo\\gt_data'
    graphs_path = 'Z:\\Cristina\\Section3\\breast_MR_pipeline\\graphs'
    triangulation_path = 'Z:\\Cristina\\Section3\\breast_MR_pipeline\\triangulations'

    if not os.path.exists(triangulation_path):
        os.mkdir(triangulation_path)
        
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
        
        #############################
        ###### 3) load graph object into memory
        #############################
        with gzip.open( os.path.join(graphs_path,'{}_{}_{}_lesion_querygraph.pklz'.format(str(lesion_id),fStudyID.zfill(4),AccessionN)), 'rb') as f:
            lesion_delaunay = pickle.load(f)
        
        # read only one bilateral volume for left/right splitting
        bilateral_filename = '{}_{}_{}_{}@'.format(str(lesion_id),fStudyID.zfill(4),AccessionN,DynSeries_nums[1] )
        bilateral_filepath = glob.glob(os.path.join( processed_path,bilateral_filename+'*'+'_mc.mha' ))[0]
        bilateral_Vol = sitk.GetArrayFromImage(sitk.Cast(sitk.ReadImage(bilateral_filepath),sitk.sitkFloat32)) 
        # get original No of slices
        nslices = bilateral_Vol.shape[0]
        lefts = int(round(nslices/2))
        rights = nslices - lefts
        
        print "Reading gt_lesions mask..."
        gt_lesion_filename = '{}_{}_{}.mha'.format(str(lesion_id),fStudyID.zfill(4),AccessionN)
        gt_lesion_filepath = os.path.join(gt_path,gt_lesion_filename)
        gt_lesion = sitk.GetArrayFromImage(sitk.Cast( sitk.ReadImage(gt_lesion_filepath), sitk.sitkFloat32)) 

        print "Pinpointing lesion slice..." 
        centroid_info = lesion_coords['lesion_centroid_ijk']
        centroid = [int(c) for c in centroid_info[1:-1].split(',')]
        
        # based onsplit left/right and recalculate centroid
        lesions = centroid[2]
        if(lesions > lefts): # lesion is on the right
            centroid[2] = centroid[2]-lefts
            # compute gt_lesion based on split left/right
            side_gt_lesion = gt_lesion[rights:,:,:]
        else:
            side_gt_lesion = gt_lesion[:lefts,:,:]
 
        ####################
        #### 4) masked gt mx_side_gt to sample from mriVols
        ### And compute SER Volume
        ####################
        # masked gt mx_side_gt
        mx_side_gt = ma.masked_array(side_gt_lesion, mask=side_gt_lesion==0)
        masked_side_gt = ma.filled(mx_side_gt, fill_value=0.0)
        
        outlines_side_gt = find_contours(masked_side_gt[centroid[2],:,:], 0)
        coords_side_gt = approximate_polygon( np.concatenate(outlines_side_gt), tolerance=2)
        
        mx_query = np.zeros(mriVols[0].shape)
        ext_x = [int(ex) for ex in [np.min(coords_side_gt[:,1])-10,np.max(coords_side_gt[:,1])+10] ] # old way int(centroid[0]-pix_x/2),int(centroid[0]+pix_x/2)
        ext_y = [int(ey) for ey in [np.min(coords_side_gt[:,0])-10,np.max(coords_side_gt[:,0])+10] ] # int(centroid[1]-pix_y/2),int(centroid[1]+pix_y/2)
        mx_query[int(centroid[2]), ext_y[0]:ext_y[1], ext_x[0]:ext_x[1]] = 1
        
        mask_queryVols = []
        onlyROI = []
        for k in range(5):
            mx = ma.masked_array(mriVols[k], mask=mx_query==0)
            print "masked lesionVol_%i, lesion mean SI/enhancement = %f" % (k, mx.mean())
            mask_queryVols.append( ma.filled(mx, fill_value=None) )
            onlyROI.append( ma.compress_rowcols( mask_queryVols[k][centroid[2],:,:] ))
        
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
        nonzmax1p1SER = nonzmax1p1SER.astype(float32)
        
        # Display 
        fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(20, 12))
        for k in range(5):
            ax[0,k].imshow(mriVols[k][centroid[2],:,:], cmap=plt.cm.gray)
            ax[0,k].set_adjustable('box-forced')
            ax[0,k].set_xlabel(str(k)+'mc_'+sideB)
            
        ax[1,0].imshow(SERvol1, cmap=plt.cm.gray)
        ax[1,0].set_adjustable('box-forced')
        ax[1,0].set_xlabel('SERvol1')
        ax[1,0].axis((ext_x[0], ext_x[1], ext_y[1], ext_y[0]))
        
        ax[1,1].imshow(SERvol4, cmap=plt.cm.gray)
        ax[1,1].set_adjustable('box-forced')
        ax[1,1].set_xlabel('SERvol4')
        ax[1,1].axis((ext_x[0], ext_x[1], ext_y[1], ext_y[0]))
        
        ax[1,2].imshow(nonzmax1p1SER, cmap=plt.cm.gray)
        ax[1,2].set_adjustable('box-forced')
        ax[1,2].set_xlabel(str(np.histogram(nonzmax1p1SER)))
        ax[1,2].axis((ext_x[0], ext_x[1], ext_y[1], ext_y[0]))
        
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
        fig.savefig( os.path.join( triangulation_path,'{}_{}_{}_mriVols_{}slice_{}.pdf'.format(str(lesion_id),fStudyID.zfill(4),AccessionN,str(centroid[2]),str(cond+BenignNMaligNAnt)) ))

        ##############
        ## 5) Add lesion delaunay triangulation
        ##############
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))
        ax = axes.flatten()

        # Overlay the three images
        im = ax[0].imshow(nonzmax1p1SER, cmap=plt.cm.cool, interpolation='none', clim=[min(nonzmax1p1SER.ravel()), max(nonzmax1p1SER.ravel())])
        ax[0].axis((ext_x[0], ext_x[1], ext_y[1], ext_y[0]))
        ax[0].set_adjustable('box-forced')
        ax[0].set_xlabel('Dela. Triangulation + SER')
        
        ## Plot delaunay triangulation
        lesion_triangles = lesion_delaunay.points[lesion_delaunay.vertices]     
        lines = []
        # Triangle vertices
        A = lesion_triangles[:, 0]
        B = lesion_triangles[:, 1]
        C = lesion_triangles[:, 2]
        lines.extend(zip(A, B))
        lines.extend(zip(B, C))
        lines.extend(zip(C, A))
        lines = LineCollection(lines, color='b')
        ax[0].plot(lesion_delaunay.points[:, 0], lesion_delaunay.points[:, 1], '.r', linewidth=2)
        ax[0].add_collection(lines)
        
        # Set the colormap and norm to correspond to the data for which
        divider = make_axes_locatable(ax[0])
        caxSER = divider.append_axes("right", size="20%", pad=0.05)
        plt.colorbar(im, cax=caxSER)
        
        #########################
        # 6) Calculate triangulation faces, compute faces centroid, use as new triangulation points
        #########################
        trianF = []
        for verT in lesion_triangles:
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
        # get original position of points
        pos = dict(zip(nodes,pts))
        # mapping from vertices to nodes
        m = dict(enumerate(nodes)) 
        
        #Create a graph
        lesionG = nx.Graph() 
        for i in range(len(pts)):
            # add position as node attributes
            lesionG.add_node(i, pos=pos[i])
        
        print "Average triangle weight" 
        for i in range(trianglFaces_delaunay.nsimplex):
            # add edges with weigth attributes
            wp1 = np.max([nodew[m[trianglFaces_delaunay.vertices[i,0]]], nodew[m[trianglFaces_delaunay.vertices[i,1]]]])
            lesionG.add_edge( m[trianglFaces_delaunay.vertices[i,0]], m[trianglFaces_delaunay.vertices[i,1]], weight = wp1 )
            wp2 = np.max([nodew[m[trianglFaces_delaunay.vertices[i,1]]], nodew[m[trianglFaces_delaunay.vertices[i,2]]]])
            lesionG.add_edge( m[trianglFaces_delaunay.vertices[i,1]], m[trianglFaces_delaunay.vertices[i,2]], weight = wp2 )
            wp3 = np.max([nodew[m[trianglFaces_delaunay.vertices[i,2]]], nodew[m[trianglFaces_delaunay.vertices[i,0]]]])
            lesionG.add_edge( m[trianglFaces_delaunay.vertices[i,2]], m[trianglFaces_delaunay.vertices[i,0]], weight = wp3 )
            print(np.mean([wp1, wp2, wp3]))                       

        # draw
        ax[1].imshow(mriVols[4][centroid[2],:,:], cmap=plt.cm.gray)
        ax[1].set_adjustable('box-forced')
        ax[1].axis((ext_x[0], ext_x[1], ext_y[1], ext_y[0]))
        nx.draw_networkx(lesionG, pos, ax=ax[1], with_labels=False, node_color='c',node_size=10, edge_color='b', width=1.5)
        ax[1].set_xlabel('4th postc mriVol + FacesTriangulated lesion graph')  

        #############################
        ###### 5) Calculate minimum spanning Tree (MST)
        #############################
        MST = nx.minimum_spanning_edges( lesionG, data=True)
        edgelist = list(MST)
        
        #Convert minimum spanning tree into graph to be able to depth searched
        MST_lesionG = nx.from_edgelist(edgelist)
        for i in range(len(pts)):
            # add position as node attributes
            MST_lesionG.node[i]['pos']=pos[i]
            
        print "all edges"
        print MST_lesionG.edges(data=True)
        SER_nodeweights = [d['weight'] for (u,v,d) in MST_lesionG.edges(data=True)]
        
        # remove edges with very high weight
#        SER_smallw_lesionG = MST_lesionG
#        for node in MST_lesionG.nodes():
#            edges = MST_lesionG.edges(node, data=True)
#            print edges
#            if len(edges) > 0: #some nodes have zero edges going into it
#                ave_weight = np.mean([edge[2]['weight'] for edge in edges])
#                print ave_weight
#                for edge in edges:
#                    if edge[2]['weight'] >= ave_weight:
#                        SER_smallw_lesionG.remove_edge(edge[0], edge[1])
#        # make grpah (removes many edges)                        
#        SER_smallw_lesionG = nx.from_edgelist(SER_smallw_edgelist)
        # to restablish node positions
        #pts = [a['pos'] for (n,a) in MST_lesionG.nodes(data=True)]

        nxg = nx.draw_networkx_edges(MST_lesionG, pos, ax=ax[2], edge_color=SER_nodeweights, edge_cmap=plt.cm.inferno, 
                                     edge_vmin=min(SER_nodeweights),edge_vmax=max(SER_nodeweights), width=1.5)
        v = np.linspace(min(SER_nodeweights), max(SER_nodeweights), 10, endpoint=True)
        
        divider = make_axes_locatable(ax[2])
        caxEdges = divider.append_axes("right", size="20%", pad=0.05)
        plt.colorbar(nxg, cax=caxEdges, ticks=v)
        
        ax[2].imshow(nonzmax1p1SER, cmap=plt.cm.cool, interpolation='none', clim=[min(nonzmax1p1SER.ravel()), max(nonzmax1p1SER.ravel())])
        ax[2].axis((ext_x[0], ext_x[1], ext_y[1], ext_y[0]))
        ax[2].set_xlabel('MST FacesTriangulated SER edge weighted ')

        # Overlay with the mri image
        nxg = nx.draw_networkx_edges(MST_lesionG, pos, ax=ax[3], edge_color=SER_nodeweights, edge_cmap=plt.cm.inferno, 
                                     edge_vmin=min(SER_nodeweights),edge_vmax=max(SER_nodeweights), width=1.5)
        nx.draw_networkx_nodes(MST_lesionG, pos, ax=ax[3], node_size=25, node_shape='.', node_color='g', with_labels=False)
        ax[3].imshow(mriVols[4][centroid[2],:,:], cmap=plt.cm.gray)
        ax[3].axis((ext_x[0], ext_x[1], ext_y[1], ext_y[0]))
        ax[3].set_adjustable('box-forced')
        ax[3].set_xlabel('MST FacesTriangulated lesion graph + SER edge weighted')
        
        divider = make_axes_locatable(ax[3])
        caxEdges = divider.append_axes("right", size="20%", pad=0.05)
        plt.colorbar(nxg, cax=caxEdges, ticks=v)
        
        #show and save  
        plt.show(block=False)                    
        fig.savefig( os.path.join( triangulation_path,'{}_{}_{}_DelauTriangFaces_{}slice_{}.pdf'.format(str(lesion_id),fStudyID.zfill(4),AccessionN,str(centroid[2]),str(cond+BenignNMaligNAnt)) ))
        plt.close("all")
        

        ####################
        # save all slices collection of query graph triangulations
        ####################
        nxgraph_lesionG   = gzip.open(os.path.join(graphs_path,'{}_{}_{}_{}_FacesTriang_lesion_nxgraph.pklz'.format(str(lesion_id),fStudyID.zfill(4),AccessionN,sideB)), 'wb')
        pickle.dump(lesionG, nxgraph_lesionG, protocol=pickle.HIGHEST_PROTOCOL)
        nxgraph_lesionG.close()
        
        nxgraph_MST_lesionG   = gzip.open(os.path.join(graphs_path,'{}_{}_{}_{}_MST_lesion_nxgraph.pklz'.format(str(lesion_id),fStudyID.zfill(4),AccessionN,sideB)), 'wb')
        pickle.dump(MST_lesionG, nxgraph_MST_lesionG, protocol=pickle.HIGHEST_PROTOCOL)
        nxgraph_MST_lesionG.close()
        
        # TO LOAD
#        with gzip.open(os.path.join(graphs_path,'{}_{}_{}_{}_FacesTriang_lesion_nxgraph.pklz'.format(str(lesion_id),fStudyID.zfill(4),AccessionN,sideB)), 'rb') as fu:
#            lesion_delaunay = pickle.load(fu)
#        
#        with gzip.open(os.path.join(graphs_path,'{}_{}_{}_{}_MST_lesion_nxgraph.pklz'.format(str(lesion_id),fStudyID.zfill(4),AccessionN,sideB)), 'rb') as fu:
#            lesion_MST = pickle.load(fu)



        ## next line
        line = file_ids.readline()       
        
    return
    

if __name__ == '__main__':
    run_nxGraphs()    
    
    