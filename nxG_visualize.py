# -*- coding: utf-8 -*-
"""
Created on Thu Oct 06 08:59:20 2016

@author: DeepLearning
"""

  
import pandas as pd
import os
import os.path, sys
import shutil
import glob
import numpy as np
import SimpleITK as sitk

import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
sns.set(color_codes=True)

import scipy.sparse as sp
from sklearn.datasets import load_mlcomp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

from matplotlib.collections import LineCollection
import six.moves.cPickle as pickle
import gzip

# to query local databse
from mylocalbase import localengine
from sqlalchemy.orm import sessionmaker
import mylocaldatabase_new
from parse_probmaps import querylocalDatabase_wRad

import numpy.ma as ma
from sklearn.manifold import TSNE

from scipy.spatial import Delaunay
import networkx as nx
from networkx.algorithms import centrality


def plot_embedding(X, y, pddata, title=None,  plotextra=False):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib import offsetbox
    from matplotlib.offsetbox import TextArea, AnnotationBbox
    
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(12, 8)) 
    ax = plt.subplot(111)
    # process labels 
    classes = [str(c) for c in np.unique(y)]
    colors=plt.cm.rainbow(np.linspace(0,1,len(classes)))
    c_patchs = []
    for k in range(len(classes)):
         c_patchs.append(mpatches.Patch(color=colors[k], label=classes[k]))
    plt.legend(handles=c_patchs)
    
    for i in range(X.shape[0]):
        for k in range(len(classes)):
            if str(y[i])==classes[k]: 
                colori = colors[k] 
        plt.text(X[i, 0], X[i, 1], str(y[i]), color=colori,
                 fontdict={'weight': 'bold', 'size': 10})

        # only print thumbnails with matplotlib > 1.0
        if(plotextra):
            shown_images = np.array([[1., 1.]])  # just something big
            for i in range(X.shape[0]):
                dist = np.sum((X[i] - shown_images) ** 2, 1)
                if np.min(dist) < 4e-3:
                    # don't show points that are too close
                    continue
                shown_images = np.r_[shown_images, [X[i]]]
                offsetbox = TextArea(pddata['type'][i], minimumdescent=False)
                extraLabelbox = AnnotationBbox(offsetbox, X[i], arrowprops=dict(arrowstyle="->"))
                ax.add_artist(extraLabelbox)
    
    ax.set_xlim(-0.1,1.1)
    ax.set_ylim(-0.1,1.1)
    if title is not None:
        plt.title(title)

def plot_embedding_showNN(tsne_id, X_tsne, y_tsne, lesion_id, lesion_MST, mriLesionSlice, nxG_name, processed_path, graphs_path, title=None, plotextra=True):
    '''Scale and visualize the embedding vectors
    version _showG requires additional inputs like lesion_id and corresponding mriVol    
    '''
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import ConnectionPatch
    import scipy.spatial as spatial

    ########################################
    pts2lesionid = open("listofprobmaps.txt","r") 
    pts2lesionid.seek(0)
    line_offset = []
    offset = 0
    for line in pts2lesionid:
        line_offset.append(offset)
        offset += len(line)
    pts2lesionid.seek(0)
   
    x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
    X_tsne = (X_tsne - x_min) / (x_max - x_min)

    figTSNE = plt.figure(figsize=(16, 16))
    G = gridspec.GridSpec(4, 4)
    # for tsne
    ax1 = plt.subplot(G[0:3, 0:3])
    # fo lesion id graph
    ax2 = plt.subplot(G[0,3])
    # plot for neighbors
    ax3 = plt.subplot(G[1,3])
    ax4 = plt.subplot(G[2,3])
    ax5 = plt.subplot(G[3,3])
    ax6 = plt.subplot(G[3,2])
    ax7 = plt.subplot(G[3,1])
    ax8 = plt.subplot(G[3,0])
    axes = [ax3,ax4,ax5,ax6,ax7,ax8]
    #
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    
    # process labels 
    classes = [str(c) for c in np.unique(y_tsne)]
    colors=plt.cm.rainbow(np.linspace(0,1,len(classes)))
    c_patchs = []
    for k in range(len(classes)):
         c_patchs.append(mpatches.Patch(color=colors[k], label=classes[k]))
    ax1.legend(handles=c_patchs, bbox_to_anchor=(1, 1))
    ax1.grid(False)    
    
    ## plot tsne
    for i in range(X_tsne.shape[0]):
        for k in range(len(classes)):
            if str(y_tsne[i])==classes[k]: 
                colori = colors[k] 
        ax1.text(X_tsne[i, 0], X_tsne[i, 1], str(y_tsne[i]), color=colori,
                 fontdict={'weight': 'bold', 'size': 8})     
                 
    # us e ConnectorPatch is useful when you want to connect points in different axes
    con1 = ConnectionPatch(xyA=(0,1), xyB=X_tsne[tsne_id], coordsA='axes fraction', coordsB='data',
            axesA=ax2, axesB=ax1, arrowstyle="simple",connectionstyle='arc3')
    ax2.add_artist(con1)                   

    posV = np.asarray([p['pos'] for (v,p) in lesion_MST.nodes(data=True)])
    yminG, xminG = np.min(posV, axis=0)
    ymaxG, xmaxG = np.max(posV, axis=0)    
    ext_x = [int(ex) for ex in [xminG-5,xmaxG+5] ] # old way int(centroid[0]-pix_x/2),int(centroid[0]+pix_x/2)
    ext_y = [int(ey) for ey in [yminG-5,ymaxG+5] ] # int(centroid[1]-pix_y/2),int(centroid[1]+pix_y/2)

    ###### Examine SER distributions acording to lesion type
    ax2.imshow(mriLesionSlice, cmap=plt.cm.gray)
    ax2.set_adjustable('box-forced')   
            
    lesion_SER_edgesw = [d['weight'] for (u,v,d) in lesion_MST.edges(data=True)]
    nx.draw_networkx_edges(lesion_MST, posV, ax=ax2, edge_color=lesion_SER_edgesw, edge_cmap=plt.cm.inferno,
                                 edge_vmin=-0.01, edge_vmax=2.5, width=1.5)
    nx.draw_networkx_nodes(lesion_MST, posV, ax=ax2, node_size=25, node_shape='.', node_color='c', with_labels=False)
    ax2.axis((ext_y[0], ext_y[1], ext_x[1], ext_x[0]))   
    ax2.set_title('lesion_MST_tsne_id_'+str(tsne_id))
    ax2.set_xlabel(nxG_name)
    ax2.grid(False)    
           
    # only add neighbors if plotextra
    if(plotextra):
        # add lesion_id graph
        #png_lesion = read_png(glob.glob(os.path.join(SER_edgesw_path,str(lesion_id)+'*.png'))[0])       
        # Find closest neighborhs and plot
        X_embedding_tree = spatial.cKDTree(X_tsne, compact_nodes=True)
        # This finds the index of all points within distance 0.1 of embedded point X_tsne[lesion_id]
        NN_embedding_indx_list = X_embedding_tree.query_ball_point(X_tsne[tsne_id], 0.05)
        print NN_embedding_indx_list
        NN_embedding_indx = [knn for knn in NN_embedding_indx_list if knn != tsne_id]
        k_nn = min(6,len(NN_embedding_indx))
        # plot knn embedded poitns
        for k in range(k_nn):
            k_nn_line = NN_embedding_indx[k]
            
            ######  Querying Research database for clinical, pathology, radiology data
            # Now, to skip to line n (with the first line being line 0), just do
            pts2lesionid.seek(line_offset[k_nn_line])
            k_nn_lesion_id, fStudyID, AccessionN, sideB, a,b,v,d,e  = pts2lesionid.readline().split('_')
            print "loading knn %i, k_nn_lesion_id=%s, fStudyID=%s, AccessionN=%s, sideB=%s" % (k, k_nn_lesion_id, fStudyID, AccessionN, sideB) 
            
            with gzip.open( os.path.join(graphs_path,'{}_{}_{}_{}_MST_lesion_nxgraph.pklz'.format(str(k_nn_lesion_id),fStudyID.zfill(4),AccessionN,sideB)), 'rb') as f:
                k_nn_lesion_MST = pickle.load(f)
                
            # us e ConnectorPatch is useful when you want to connect points in different axes
            conknn = ConnectionPatch(xyA=(0,1), xyB=X_tsne[k_nn_line], coordsA='axes fraction', coordsB='data',
                    axesA=axes[k], axesB=ax1, arrowstyle="simple",connectionstyle='arc3')
            axes[k].add_artist(conknn) 
            
            ## query database
            cond, BenignNMaligNAnt, Diagnosis, casesFrame, MorNMcase, lesion_coords = querylocalDatabase_wRad(k_nn_lesion_id, verbose=False)            

            # get dynmic series info
            DynSeries_id = MorNMcase['DynSeries_id']
            precontrast_id = int(str(DynSeries_id[1:])) #s600, [1:] remove the 's'
            DynSeries_nums = [str(n) for n in range(precontrast_id,precontrast_id+5)]
            #the output mha:lesionid_patientid_access#_series#@acqusionTime.mha
            DynSeries_filename = '{}_{}_{}_{}'.format(str(k_nn_lesion_id),fStudyID.zfill(4),AccessionN,DynSeries_nums[4] )
            #write log if mha file not exist             
            glob_result = glob.glob(os.path.join(processed_path,DynSeries_filename+'*')) #'*':do not to know the exactly acquistion time
            if glob_result != []:
                filename = glob_result[0]
            # add side info from the side of the lesion
            filename_wside = filename[:-4]+'_{}.mha'.format(sideB)
            mriVol = sitk.GetArrayFromImage(sitk.Cast(sitk.ReadImage(filename_wside),sitk.sitkFloat32)) 
            # read only one bilateral volume for left/right splitting
            bilateral_filename = '{}_{}_{}_{}@'.format(str(k_nn_lesion_id),fStudyID.zfill(4),AccessionN,DynSeries_nums[1] )
            bilateral_filepath = glob.glob(os.path.join( processed_path,bilateral_filename+'*'+'_mc.mha' ))[0]
            bilateral_Vol = sitk.GetArrayFromImage(sitk.Cast(sitk.ReadImage(bilateral_filepath),sitk.sitkFloat32)) 

            # get original No of slices
            nslices = bilateral_Vol.shape[0]
            lefts = int(round(nslices/2))
            print "Pinpointing lesion slice..." 
            centroid_info = lesion_coords['lesion_centroid_ijk']
            centroid = [int(c) for c in centroid_info[1:-1].split(',')]
            #based onsplit left/right and recalculate centroid
            lesions = centroid[2]
            if(lesions > lefts): # lesion is on the right
                centroid[2] = centroid[2]-lefts
    
            print "masking lesion ROI"
            posV = np.asarray([p['pos'] for (v,p) in k_nn_lesion_MST.nodes(data=True)])
            yminG, xminG = np.min(posV, axis=0)
            ymaxG, xmaxG = np.max(posV, axis=0)
            
            mx_query = np.zeros(mriVol.shape)
            ext_x = [int(ex) for ex in [xminG-5,xmaxG+5] ] # old way int(centroid[0]-pix_x/2),int(centroid[0]+pix_x/2)
            ext_y = [int(ey) for ey in [yminG-5,ymaxG+5] ] # int(centroid[1]-pix_y/2),int(centroid[1]+pix_y/2)
            mx_query[int(centroid[2]), ext_y[0]:ext_y[1], ext_x[0]:ext_x[1]] = 1

            axes[k].imshow(mriVol[centroid[2],:,:], cmap=plt.cm.gray)
            axes[k].set_adjustable('box-forced')   
            
            ###### Examine SER distributions acording to lesion type
            lesion_SER_edgesw = [d['weight'] for (u,v,d) in k_nn_lesion_MST.edges(data=True)]
            nx.draw_networkx_edges(k_nn_lesion_MST, posV, ax=axes[k], edge_color=lesion_SER_edgesw, edge_cmap=plt.cm.inferno,
                                         edge_vmin=-0.01, edge_vmax=2.5, width=1.5)
            nx.draw_networkx_nodes(k_nn_lesion_MST, posV, ax=axes[k], node_size=25, node_shape='.', node_color='c', with_labels=False)
            axes[k].axis((ext_y[0], ext_y[1], ext_x[1], ext_x[0]))       
            nxG_title= '{}_{}_{}_{}{}'.format(str(k_nn_lesion_id),fStudyID.zfill(4),AccessionN,cond,BenignNMaligNAnt)
            axes[k].set_title(nxG_title)
            axes[k].set_xlabel('{}'.format(Diagnosis))
            axes[k].grid(False)
            #reset
            pts2lesionid.seek(0)
               
    if title is not None:
        plt.title(title)
    
    plt.tight_layout()
    return figTSNE
        

def run_visualizeTSNE_SERedgew(saveFigs=False):
    # Get Root folder ( the directory of the script being run)
    path_rootFolder = os.path.dirname(os.path.abspath(__file__))
    os.chdir(path_rootFolder) 
    
    processed_path = 'Y:\\Hongbo\\processed_data'
    graphs_path = 'Z:\\Cristina\\Section3\\breast_MR_pipeline\\graphs'
    SER_edgesw_path = 'Z:\\Cristina\\Section3\\breast_MR_pipeline\\lesion_SER_edgesw'
    
    if not os.path.exists(SER_edgesw_path):
        os.mkdir(SER_edgesw_path)
        
    ########################################
    ## for visualization load the t-SNE mapping of lesions
    ########################################
    # to load SERw matrices for all lesions
    with gzip.open(os.path.join(SER_edgesw_path,'allLesions_10binsizenormSERcounts_probC_x.pklz'), 'rb') as fu:
        normSERcounts = pickle.load(fu)
    
    with gzip.open(os.path.join(SER_edgesw_path,'allLesions_10binsizedatanormSERcounts_probC_x.pklz'), 'rb') as fu:
        datanormSERcounts = pickle.load(fu)

    # set up some parameters and define labels
    X = np.asarray(normSERcounts)
    print"Input data to t-SNE is mxn-dimensional with m = %i discretized SER bins" % X.shape[1]
    print"Input data to t-SNE is mxn-dimensional with n = %i cases" % X.shape[0]
    y =  datanormSERcounts['class'].values
    y2 = datanormSERcounts['type'].values

    tsne = TSNE(n_components=2, perplexity=12, early_exaggeration=8, learning_rate=700, 
                     init='pca', random_state=0, verbose=2, method='exact')      
    X_tsne = tsne.fit_transform(X)
    y_tsne = y #y2+y
    
    ########################################
    file_ids = open("listofprobmaps.txt","r") 
    file_ids.seek(0)
    line = file_ids.readline()
    
    tsne_id = 0
    while ( line ) : 
        # Get the line: id  CAD study #	Accession #	Date of Exam	Pathology Diagnosis 
        print(line)
        fileline = line.split('_')
        try:
            lesion_id = int(fileline[0] )
            fStudyID = fileline[1] 
            AccessionN = fileline[2]  
            sideB = fileline[3]
        
            #############################
            ###### 1) Querying Research database for clinical, pathology, radiology data
            #############################
            cond, BenignNMaligNAnt, Diagnosis, casesFrame, MorNMcase, lesion_coords = querylocalDatabase_wRad(lesion_id, verbose=True)            
            
            #############################
            ###### 2) load DEL and MST graph object into memory
            #############################
            with gzip.open( os.path.join(graphs_path,'{}_{}_{}_{}_FacesTriang_lesion_nxgraph.pklz'.format(str(lesion_id),fStudyID.zfill(4),AccessionN,sideB)), 'rb') as f:
                lesion_DEL = pickle.load(f)
            
            with gzip.open( os.path.join(graphs_path,'{}_{}_{}_{}_MST_lesion_nxgraph.pklz'.format(str(lesion_id),fStudyID.zfill(4),AccessionN,sideB)), 'rb') as f:
                lesion_MST = pickle.load(f)
           
            #############################
            ###### 3) Accesing mc images (last post contast) for visualization purposes
            #############################
            # get dynmic series info
            DynSeries_id = MorNMcase['DynSeries_id']
            precontrast_id = int(str(DynSeries_id[1:])) #s600, [1:] remove the 's'
            DynSeries_nums = [str(n) for n in range(precontrast_id,precontrast_id+5)]
            #the output mha:lesionid_patientid_access#_series#@acqusionTime.mha
            DynSeries_filename = '{}_{}_{}_{}'.format(str(lesion_id),fStudyID.zfill(4),AccessionN,DynSeries_nums[4] )
            #write log if mha file not exist             
            glob_result = glob.glob(os.path.join(processed_path,DynSeries_filename+'*')) #'*':do not to know the exactly acquistion time
            if glob_result != []:
                filename = glob_result[0]
            # add side info from the side of the lesion
            filename_wside = filename[:-4]+'_{}.mha'.format(sideB)
            mriVol = sitk.GetArrayFromImage(sitk.Cast(sitk.ReadImage(filename_wside),sitk.sitkFloat32)) 
            
            # read only one bilateral volume for left/right splitting
            bilateral_filename = '{}_{}_{}_{}@'.format(str(lesion_id),fStudyID.zfill(4),AccessionN,DynSeries_nums[1] )
            bilateral_filepath = glob.glob(os.path.join( processed_path,bilateral_filename+'*'+'_mc.mha' ))[0]
            bilateral_Vol = sitk.GetArrayFromImage(sitk.Cast(sitk.ReadImage(bilateral_filepath),sitk.sitkFloat32)) 

            # get original No of slices
            nslices = bilateral_Vol.shape[0]
            lefts = int(round(nslices/2))
            
            print "Pinpointing lesion slice..." 
            centroid_info = lesion_coords['lesion_centroid_ijk']
            centroid = [int(c) for c in centroid_info[1:-1].split(',')]
            #based onsplit left/right and recalculate centroid
            lesions = centroid[2]
            if(lesions > lefts): # lesion is on the right
                centroid[2] = centroid[2]-lefts
    
            print "masking lesion ROI"
            posV = np.asarray([p['pos'] for (v,p) in lesion_DEL.nodes(data=True)])
            yminG, xminG = np.min(posV, axis=0)
            ymaxG, xmaxG = np.max(posV, axis=0)
            
            mx_query = np.zeros(mriVol.shape)
            ext_x = [int(ex) for ex in [xminG-5,xmaxG+5] ] # old way int(centroid[0]-pix_x/2),int(centroid[0]+pix_x/2)
            ext_y = [int(ey) for ey in [yminG-5,ymaxG+5] ] # int(centroid[1]-pix_y/2),int(centroid[1]+pix_y/2)
            mx_query[int(centroid[2]), ext_y[0]:ext_y[1], ext_x[0]:ext_x[1]] = 1
            
            #############################
            ###### 3) Examine SER distributions acording to lesion type
            ############################# 
            # For now, using undirected graphs with node weigths corresponding to SER values 
            # at node linkages (pixel locations)
            # in this case, iterate through all edge weigths (SER values in lesion) and summarize in histogram
            lesion_SER_edgesw = [d['weight'] for (u,v,d) in lesion_MST.edges(data=True)]
           
            # selection of mriROI and calculate extension
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 20))
            ax.imshow(mriVol[centroid[2],:,:], cmap=plt.cm.gray)
            ax.set_adjustable('box-forced')
            nxG_name = '{}_{}_{}_MST_nxG_{}{}_{}'.format(str(lesion_id),fStudyID.zfill(4),AccessionN,cond,BenignNMaligNAnt,Diagnosis)
            ax.set_xlabel(nxG_name)
            ax.axis((ext_y[0], ext_y[1], ext_x[1], ext_x[0]))
            ax.grid(False)
            # add nxG delaunay
            nx.draw_networkx_edges(lesion_MST, posV, ax=ax, edge_color=lesion_SER_edgesw, edge_cmap=plt.cm.inferno,
                                         edge_vmin=-0.01, edge_vmax=2.5, width=5)
            nx.draw_networkx_nodes(lesion_MST, posV, ax=ax, node_size=120, node_shape='.', node_color='c', with_labels=False)
            
            #show and save
            if(saveFigs):
                fig.savefig( os.path.join( SER_edgesw_path, nxG_name+'.png' ), bbox_inches='tight') 
            plt.close()
            
            #############################
            ###### 4) Examine and plot TSNE with KNN neighbor graphs in a radius of tnse embedding = 0.1
            #############################  
            mriLesionSlice = mriVol[centroid[2],:,:]                             
            figTSNE = plot_embedding_showNN(tsne_id, X_tsne, y_tsne, lesion_id, lesion_MST, mriLesionSlice, nxG_name, processed_path, graphs_path, title=None, plotextra=True)   
            #show and save
            if(saveFigs):
                figTSNE.savefig( os.path.join(SER_edgesw_path,'lesion_id_{}_TSNE_{}{}_{}.pdf'.format(str(lesion_id),cond,BenignNMaligNAnt,Diagnosis)), bbox_inches='tight') 
            plt.close()

            ## next line
            line = file_ids.readline()  
            tsne_id+=1
     
        except:
            return 
            
    return   


def run_visualizeTSNE_nxGwSER(saveFigs=False):
    # Get Root folder ( the directory of the script being run)
    path_rootFolder = os.path.dirname(os.path.abspath(__file__))
    os.chdir(path_rootFolder) 
    
    processed_path = 'Y:\\Hongbo\\processed_data'
    graphs_path = 'Z:\\Cristina\\Section3\\breast_MR_pipeline\\graphs'
    SER_edgesw_path = 'Z:\\Cristina\\Section3\\breast_MR_pipeline\\lesion_SER_edgesw'
    nxGfeatures_path = 'Z:\\Cristina\\Section3\\breast_MR_pipeline\\lesion_nxGfeatures'

    if not os.path.exists(nxGfeatures_path):
        os.mkdir(nxGfeatures_path)
        
    ########################################
    ## for visualization load the t-SNE mapping of lesions
    ########################################
    # to load SERw matrices for all lesions
    with gzip.open(os.path.join(SER_edgesw_path,'allLesions_10binsizenormSERcounts_probC_x.pklz'), 'rb') as fu:
        normSERcounts = pickle.load(fu)
    
    with gzip.open(os.path.join(SER_edgesw_path,'allLesions_10binsizedatanormSERcounts_probC_x.pklz'), 'rb') as fu:
        datanormSERcounts = pickle.load(fu)
        
    with gzip.open(os.path.join(nxGfeatures_path,'nxGdatafeatures_allLesions_10binsize.pklz'), 'rb') as fu:
        nxGdatafeatures = pickle.load(fu)
    
    with gzip.open(os.path.join(nxGfeatures_path,'nxGnormfeatures_allLesions_10binsize.pklz'), 'rb') as fu:
        nxGnormfeatures = pickle.load(fu)
        
    # set up some parameters and define labels
    combX = np.concatenate((nxGnormfeatures, np.asarray(normSERcounts)), axis=1)
    print"Input data to t-SNE is mxn-dimensional with m = %i discretized SER and nxG features" % combX.shape[1]
    print"Input data to t-SNE is mxn-dimensional with n = %i cases" % combX.shape[0]
    y =  nxGdatafeatures['class'].values
    y2 = nxGdatafeatures['type'].values

    tsne = TSNE(n_components=2, perplexity=9, early_exaggeration=7, learning_rate=320, 
                     init='pca', random_state=0, verbose=2, method='exact')    
    X_tsne = tsne.fit_transform(combX)
    y_tsne = y #y2+y
    
    ## plot TSNE
    plot_embedding(X_tsne, y_tsne, nxGdatafeatures, title='nxGdatafeatures',  plotextra=False)
    
    ########################################
    file_ids = open("listofprobmaps.txt","r") 
    file_ids.seek(0)
    line = file_ids.readline()
    
    tsne_id = 0
    while ( line ) : 
        # Get the line: id  CAD study #	Accession #	Date of Exam	Pathology Diagnosis 
        print(line)
        fileline = line.split('_')
        try:
            lesion_id = int(fileline[0] )
            fStudyID = fileline[1] 
            AccessionN = fileline[2]  
            sideB = fileline[3]
        
            #############################
            ###### 1) Querying Research database for clinical, pathology, radiology data
            #############################
            cond, BenignNMaligNAnt, Diagnosis, casesFrame, MorNMcase, lesion_coords = querylocalDatabase_wRad(lesion_id, verbose=True)           
            
            #############################
            ###### 2) load DEL and MST graph object into memory
            #############################
            with gzip.open( os.path.join(graphs_path,'{}_{}_{}_{}_FacesTriang_lesion_nxgraph.pklz'.format(str(lesion_id),fStudyID.zfill(4),AccessionN,sideB)), 'rb') as f:
                lesion_DEL = pickle.load(f)
            
            with gzip.open( os.path.join(graphs_path,'{}_{}_{}_{}_MST_lesion_nxgraph.pklz'.format(str(lesion_id),fStudyID.zfill(4),AccessionN,sideB)), 'rb') as f:
                lesion_MST = pickle.load(f)
           
            #############################
            ###### 3) Accesing mc images (last post contast) for visualization purposes
            #############################
            # get dynmic series info
            DynSeries_id = MorNMcase['DynSeries_id']
            precontrast_id = int(str(DynSeries_id[1:])) #s600, [1:] remove the 's'
            DynSeries_nums = [str(n) for n in range(precontrast_id,precontrast_id+5)]
            #the output mha:lesionid_patientid_access#_series#@acqusionTime.mha
            DynSeries_filename = '{}_{}_{}_{}'.format(str(lesion_id),fStudyID.zfill(4),AccessionN,DynSeries_nums[4] )
            #write log if mha file not exist             
            glob_result = glob.glob(os.path.join(processed_path,DynSeries_filename+'*')) #'*':do not to know the exactly acquistion time
            if glob_result != []:
                filename = glob_result[0]
            # add side info from the side of the lesion
            filename_wside = filename[:-4]+'_{}.mha'.format(sideB)
            mriVol = sitk.GetArrayFromImage(sitk.Cast(sitk.ReadImage(filename_wside),sitk.sitkFloat32)) 
            
            # read only one bilateral volume for left/right splitting
            bilateral_filename = '{}_{}_{}_{}@'.format(str(lesion_id),fStudyID.zfill(4),AccessionN,DynSeries_nums[1] )
            bilateral_filepath = glob.glob(os.path.join( processed_path,bilateral_filename+'*'+'_mc.mha' ))[0]
            bilateral_Vol = sitk.GetArrayFromImage(sitk.Cast(sitk.ReadImage(bilateral_filepath),sitk.sitkFloat32)) 

            # get original No of slices
            nslices = bilateral_Vol.shape[0]
            lefts = int(round(nslices/2))
            
            print "Pinpointing lesion slice..." 
            centroid_info = lesion_coords['lesion_centroid_ijk']
            centroid = [int(c) for c in centroid_info[1:-1].split(',')]
            #based onsplit left/right and recalculate centroid
            lesions = centroid[2]
            if(lesions > lefts): # lesion is on the right
                centroid[2] = centroid[2]-lefts
    
            print "masking lesion ROI"
            posV = np.asarray([p['pos'] for (v,p) in lesion_DEL.nodes(data=True)])
            yminG, xminG = np.min(posV, axis=0)
            ymaxG, xmaxG = np.max(posV, axis=0)
            
            mx_query = np.zeros(mriVol.shape)
            ext_x = [int(ex) for ex in [xminG-5,xmaxG+5] ] # old way int(centroid[0]-pix_x/2),int(centroid[0]+pix_x/2)
            ext_y = [int(ey) for ey in [yminG-5,ymaxG+5] ] # int(centroid[1]-pix_y/2),int(centroid[1]+pix_y/2)
            mx_query[int(centroid[2]), ext_y[0]:ext_y[1], ext_x[0]:ext_x[1]] = 1
        
            
            #############################
            ###### 4) Examine and plot TSNE with KNN neighbor graphs in a radius of tnse embedding = 0.1
            #############################  
            mriLesionSlice = mriVol[centroid[2],:,:]           
            nxG_name = '{}_{}_{}_MST_nxG_{}{}_{}'.format(str(lesion_id),fStudyID.zfill(4),AccessionN,cond,BenignNMaligNAnt,Diagnosis)
                  
            figTSNE = plot_embedding_showNN(tsne_id, X_tsne, y_tsne, lesion_id, lesion_MST, mriLesionSlice, nxG_name, processed_path, graphs_path, title=None, plotextra=True)   
            #show and save
            if(saveFigs):
                figTSNE.savefig( os.path.join(nxGfeatures_path,'lesion_id_{}_TSNE_nxGwSER_{}{}_{}.pdf'.format(str(lesion_id),cond,BenignNMaligNAnt,Diagnosis)), bbox_inches='tight') 
            plt.close()

            ## next line
            line = file_ids.readline()  
            tsne_id+=1
     
        except:
            return 
            
    return   


if __name__ == '__main__':
    """
    usage:
    
    from nxG_algorithms import *
    from nxG_visualize import *
    
    # to run for lesions in simple SERedgew TSNE, and visualize with closest neighbors
    run_visualizeTSNE_SERedgew(saveFigs=True)
    
    ########################################
    ## for visualization load the t-SNE mapping of lesions
    ########################################
    # to load SERw matrices for all lesions
    SER_edgesw_path = 'Z:\\Cristina\\Section3\\breast_MR_pipeline\\lesion_SER_edgesw'
    with gzip.open(os.path.join(SER_edgesw_path,'allLesions_10binsizenormSERcounts_probC_x.pklz'), 'rb') as fu:
        normSERcounts = pickle.load(fu)
    
    with gzip.open(os.path.join(SER_edgesw_path,'allLesions_10binsizedatanormSERcounts_probC_x.pklz'), 'rb') as fu:
        datanormSERcounts = pickle.load(fu)

    # set up some parameters and define labels
    X = np.asarray(normSERcounts)
    print"Input data to t-SNE is mxn-dimensional with m = %i discretized SER bins" % X.shape[1]
    print"Input data to t-SNE is mxn-dimensional with n = %i cases" % X.shape[0]
    y =  datanormSERcounts['class'].values
    y2 = datanormSERcounts['type'].values

    tsne = TSNE(n_components=2, perplexity=12, early_exaggeration=8, learning_rate=700, 
                     init='pca', random_state=0, verbose=2, method='exact')      
    X_tsne = tsne.fit_transform(X)
    y_tsne = y #y2+y
    
    ## plot TSNE
    plot_embedding(X_tsne, y, datanormSERcounts, title='datanormSERcounts',  plotextra=False)
    plot_embedding(X_tsne, y, datanormSERcounts, title='datanormSERcounts',  plotextra=True)
    
    ########################################
    # to load all nxGdatafeatures
    nxGfeatures_path = 'Z:\\Cristina\\Section3\\breast_MR_pipeline\\lesion_nxGfeatures'
    with gzip.open(os.path.join(nxGfeatures_path,'nxGdatafeatures_allLesions_10binsize.pklz'), 'rb') as fu:
        nxGdatafeatures = pickle.load(fu)
    
    with gzip.open(os.path.join(nxGfeatures_path,'nxGnormfeatures_allLesions_10binsize.pklz'), 'rb') as fu:
        nxGnormfeatures = pickle.load(fu)
        
    # to run for lesions in nxG features + SERedgew TSNE, and visualize with closest neighbors
    run_visualizeTSNE_nxGwSER(saveFigs=True)
    
    """
    