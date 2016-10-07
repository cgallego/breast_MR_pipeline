# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 10:48:24 2016

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

import scipy.sparse as sp
from sklearn.datasets import load_mlcomp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

from scipy.spatial import Delaunay
import networkx as nx
from matplotlib.collections import LineCollection
import six.moves.cPickle as pickle
import gzip

from parse_probmaps import querylocalDatabase_wRad


def run_SERcounts_probC_x(normalizedflag=False, saveFigs=False):
    # Get Root folder ( the directory of the script being run)
    path_rootFolder = os.path.dirname(os.path.abspath(__file__))
    os.chdir(path_rootFolder) 
    
    graphs_path = 'Z:\\Cristina\\Section3\\breast_MR_pipeline\\graphs'
    SER_edgesw_path = 'Z:\\Cristina\\Section3\\breast_MR_pipeline\\lesion_SER_edgesw'

    if not os.path.exists(SER_edgesw_path):
        os.mkdir(SER_edgesw_path)
        
    file_ids = open("listofprobmaps.txt","r") 
    file_ids.seek(0)
    line = file_ids.readline()
  
    dataSERcounts = pd.DataFrame({'SERcounts': [], 'class': [], 'type': []})
    SERcounts = []
    
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
            ###### 3) Examine SER distributions acording to lesion type
            ############################# 
            # For now, using undirected graphs with node weigths corresponding to SER values 
            # at node linkages (pixel locations)
            # in this case, iterate through all edge weigths (SER values in lesion) and summarize in histogram
            lesion_SER_edgesw = [d['weight'] for (u,v,d) in lesion_MST.edges(data=True)]
           
            # plot
            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
            sns.set_style("darkgrid", {"legend.frameon": True})
            pd_lesion_SER_edgesw = pd.Series(lesion_SER_edgesw, name="SER edgeweights")
            labelLesion = 'lesion_id_{}_{}{}_{}'.format(str(lesion_id),cond,BenignNMaligNAnt,Diagnosis)
            sns.distplot(pd_lesion_SER_edgesw, label=labelLesion, ax=ax[0])
            ax[0].legend()       
             
            #############################
            ###### 4) Convert Nxgraphs to edge weights discretixed by SER counts matrix
            ############################# 
            # first discretize SER values
            # 100 bins
            discrSERvals = np.linspace(-0.01,2.5,10,endpoint=True)
            discrSERcounts, discrSER_borderbins = np.histogram(lesion_SER_edgesw, bins = discrSERvals)
            
            if(normalizedflag):
                # normalize by number of edges
                discrSERcounts = discrSERcounts.astype(np.float32)/np.sum(discrSERcounts)
            
            # where discrSER_borderbins is the automatically calculated border for your bins and discrSERcounts is the population inside each bin.
            pd_lesion_SER_edgesw = pd.Series(discrSERcounts, name="")
            hist = ax[1].hist( lesion_SER_edgesw, bins = discrSERvals)
            ax[1].set_xlabel('discretized SER edgeweights (10 bin size)')
            nonzerobin=discrSERcounts!=0
            xlabels = ['%.2f' % bin if nonzerobin[i] else None for i, bin in enumerate(discrSER_borderbins[:-1])]
            ax[1].set_xlim( min(discrSER_borderbins[:-1][nonzerobin]), max(discrSER_borderbins[:-1][nonzerobin]) )
            ax[1].set_xticks(discrSER_borderbins)
            ax[1].set_xticklabels(xlabels, rotation=45, rotation_mode="anchor")
            
            xticks = ax[1].xaxis.get_major_ticks()
            for iflag in range(len(nonzerobin)):
                if not nonzerobin[iflag]:
                    xticks[iflag].label1.set_visible(False)
                    
            #show and save  
            if(saveFigs):
                fig.savefig( os.path.join( SER_edgesw_path,labelLesion+'.pdf' ) )
    
            # append to graphvector_discreSERcounts   
            # to build dataframe    
            rows = []
            index = []     
            rows.append({'SERcounts': discrSERcounts, 'class': BenignNMaligNAnt, 'type': cond})
            index.append(labelLesion)
            
            # append counts to master lists
            dataSERcounts = dataSERcounts.append( pd.DataFrame(rows, index=index) )
            SERcounts.append( list(discrSERcounts) )
           
            ## next line
            line = file_ids.readline()  
            plt.close()
        except:
            return SERcounts, dataSERcounts
        
    return SERcounts, dataSERcounts


def run_normalSERcounts_probC_x(normalizedflag=False, saveFigs=False):
    # Get Root folder ( the directory of the script being run)
    path_rootFolder = os.path.dirname(os.path.abspath(__file__))
    os.chdir(path_rootFolder) 
    
    graphs_path = 'Z:\\Cristina\\Section3\\breast_MR_pipeline\\graphs'
    SER_edgesw_path = 'Z:\\Cristina\\Section3\\breast_MR_pipeline\\lesion_SER_edgesw'

    if not os.path.exists(SER_edgesw_path):
        os.mkdir(SER_edgesw_path)
        
    file_ids = open("listofprobmaps_normals.txt","r") 
    file_ids.seek(0)
    line = file_ids.readline()
  
    normaldataSERcounts = pd.DataFrame({'SERcounts': [], 'class': [], 'type': []})
    normalSERcounts = []
    
    inormal=1
    while ( line ) : 
        # Get the line: id  CAD study #	Accession #	Date of Exam	Pathology Diagnosis 
        fileline = line.split('_')
        try:
            lesion_id = 'n'+str(inormal) 
            fStudyID = fileline[1] 
            AccessionN = fileline[2]  
            sideB = fileline[3]
        
            cond = '2ynormalfu'
            BenignNMaligNAnt = 'normal' 
            
            #############################
            ###### 1) load DEL and MST graph object into memory
            #############################        
            with gzip.open( os.path.join(graphs_path,'{}_{}_{}_{}_FacesTriang_allSlices_nxgraph.pklz'.format(str(lesion_id),fStudyID.zfill(4),AccessionN,sideB)), 'rb') as f:
                normal_MST_list = pickle.load(f)
            
            print '{}_{}_{}_{}_FacesTriang_allSlices_nxgraph.pklz'.format(str(lesion_id),fStudyID.zfill(4),AccessionN,sideB)
            
            #############################
            ###### 3) Examine SER distributions acording to lesion type
            ############################# 
            # For now, using undirected graphs with node weigths corresponding to SER values 
            # at node linkages (pixel locations)
            # in this case, iterate through all edge weigths (SER values in lesion) and summarize in histogram
            ploti = np.random.randint(0, len(normal_MST_list), 1)[0]
            lenploti = len(normal_MST_list[ploti])
            print "Selected slice %i, with %i lesions \n" % (ploti, lenploti)
            
            fig, axes = plt.subplots(nrows=int(np.sqrt(lenploti)+1), ncols=int(np.sqrt(lenploti)+2), figsize=(12, 8))
            ax = axes.flat
            sns.set_style("darkgrid", {"legend.frameon": True})
                        
            for i, islice in enumerate(normal_MST_list):
                for j, inormalnx in enumerate(islice):
                    normal_SER_edgesw = [d['weight'] for (u,v,d) in inormalnx.edges(data=True)]
           
                    #############################
                    ###### 4) Convert Nxgraphs to edge weights discretixed by SER counts matrix
                    ############################# 
                    # first discretize SER values
                    # 100 bins
                    discrSERvals = np.linspace(-0.01,2.5,10,endpoint=True)
                    discrSERcounts, discrSER_borderbins = np.histogram(normal_SER_edgesw, bins = discrSERvals)
                    
                    if(normalizedflag):
                        # normalize by number of edges
                        discrSERcounts = discrSERcounts.astype(np.float32)/np.sum(discrSERcounts)
    
                    # plot once 
                    if i==ploti and lenploti>0:
                        # where discrSER_borderbins is the automatically calculated border for your bins and discrSERcounts is the population inside each bin.
                        hist = ax[j].hist( normal_SER_edgesw, bins = discrSERvals)
                        ax[j].set_xlabel('nx'+str(j))
                        nonzerobin=discrSERcounts!=0
                        xlabels = ['%.2f' % bin if nonzerobin[ib] else None for ib, bin in enumerate(discrSER_borderbins[:-1])]
                        ax[j].set_xlim( min(discrSER_borderbins[:-1][nonzerobin]), max(discrSER_borderbins[:-1][nonzerobin]) )
                        ax[j].set_xticks(discrSER_borderbins)
                        ax[j].set_xticklabels(xlabels, rotation=45, rotation_mode="anchor")
                        
                        xticks = ax[j].xaxis.get_major_ticks()
                        for iflag in range(len(nonzerobin)):
                            if not nonzerobin[iflag]:
                                xticks[iflag].label1.set_visible(False)
                                
                        
                    # append to graphvector_discreSERcounts   
                    # to build dataframe    
                    rows = []
                    index = []     
                    rows.append({'SERcounts': discrSERcounts, 'class': BenignNMaligNAnt, 'type': cond})
                    labelLesion = 'lesion_id_{}_slice{}_nx{}'.format(str(lesion_id),i,j)
                    index.append(labelLesion)
            
                    # append counts to master lists
                    normaldataSERcounts = normaldataSERcounts.append( pd.DataFrame(rows, index=index) )
                    normalSERcounts.append( list(discrSERcounts) )
            
            #show and save
            if(saveFigs and lenploti>0):
                labelgraph = 'lesion_id_{}_slice{}_discretized_10binsize_SERw'.format(str(lesion_id),i)
                fig.savefig( os.path.join( SER_edgesw_path, labelgraph+'.pdf' ) )
    
            ## next line
            line = file_ids.readline()  
            inormal+=1
            plt.close()
        except:
            return normalSERcounts, normaldataSERcounts
        
    return normalSERcounts, normaldataSERcounts
    

# Scale and visualize the embedding vectors
def plot_embedding(X, y, pddata, title=None,  plotextra=False):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    #x_min, x_max = np.min(X, 0), np.max(X, 0)
    #X = (X - x_min) / (x_max - x_min)

    plt.figure()
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
                 fontdict={'weight': 'bold', 'size': 8})
        
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
        
        
if __name__ == '__main__':
    """
    usage:
    
    from SER_edgesw_convert2counts import *
    # to run for lesions
    SERcounts, dataSERcounts = run_SERcounts_probC_x(normalizedflag=False, saveFigs=False)
    normSERcounts, datanormSERcounts = run_SERcounts_probC_x(normalizedflag=True, saveFigs=False)   
    
    # to run for normals
    normalnormSERcounts, normaldatanormSERcounts = run_normalSERcounts_probC_x(normalizedflag=True, saveFigs=True)

    ####################
    # save all SERcounts
    ####################
    SER_edgesw_path = 'Z:\\Cristina\\Section3\\breast_MR_pipeline\\lesion_SER_edgesw'
    SERcounts_allLesions   = gzip.open(os.path.join(SER_edgesw_path,'allLesions_10binsizenormSERcounts_probC_x.pklz'), 'wb')
    pickle.dump(normSERcounts, SERcounts_allLesions, protocol=pickle.HIGHEST_PROTOCOL)
    SERcounts_allLesions.close()
    
    dataSERcounts_allLesions = gzip.open(os.path.join(SER_edgesw_path,'allLesions_10binsizedatanormSERcounts_probC_x.pklz'), 'wb')
    pickle.dump(datanormSERcounts, dataSERcounts_allLesions, protocol=pickle.HIGHEST_PROTOCOL)
    dataSERcounts_allLesions.close()
    
    # for normals
    SERcounts_allnormals   = gzip.open(os.path.join(SER_edgesw_path,'allnormals_10binsizenormSERcounts_probC_x.pklz'), 'wb')
    pickle.dump(normalnormSERcounts, SERcounts_allnormals, protocol=pickle.HIGHEST_PROTOCOL)
    SERcounts_allnormals.close()
    
    dataSERcounts_allnormals = gzip.open(os.path.join(SER_edgesw_path,'allnormals_10binsizedatanormSERcounts_probC_x.pklz'), 'wb')
    pickle.dump(normaldatanormSERcounts, dataSERcounts_allnormals, protocol=pickle.HIGHEST_PROTOCOL)
    dataSERcounts_allnormals.close()
    
    ####################
    # to load
    ####################
    # TO LOAD lesions
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
    n_neighbors = 10

    # subset only nonmasses
    Xnonmass = X[datanormSERcounts['type'].values=='nonmass',:]
    ynonmass =  datanormSERcounts['class'].values[datanormSERcounts['type'].values=='nonmass']
    datanonmass = datanormSERcounts[datanormSERcounts['type'].values=='nonmass']

    # TO LOAD normals
    SER_edgesw_path = 'Z:\\Cristina\\Section3\\breast_MR_pipeline\\lesion_SER_edgesw'
    with gzip.open(os.path.join(SER_edgesw_path,'allnormals_10binsizenormSERcounts_probC_x.pklz'), 'rb') as fu:
        normals_normSERcounts = pickle.load(fu)
    
    with gzip.open(os.path.join(SER_edgesw_path,'allnormals_10binsizedatanormSERcounts_probC_x.pklz'), 'rb') as fu:
        normals_datanormSERcounts = pickle.load(fu)
        
    # for normals need to subsample
    print "Total number of normal detections minned from probmaps = %i " % len(normals_datanormSERcounts)
    print "Average # detections minned per breast = %f " % (len(normals_datanormSERcounts)/304.0)
    normals_X = np.asarray(normals_normSERcounts)
    normals_y =  normals_datanormSERcounts['class'].values
    normals_y2 = normals_datanormSERcounts['type'].values
        
    # subsample ids (me)
    ids = np.random.randint(0, normals_X.shape[0], 100)
    sX = normals_X[ids,:]
    sy = ["N" for item in normals_y[ids]]
    sy2 = normals_y2[ids]
    sdata = normals_datanormSERcounts.iloc[ids]
    print sX.shape
    print sdata.describe()
    
    ### APPEND NONMASS AND NORMALS
    combdata = pd.concat([datanonmass,sdata])
    combX = np.concatenate( (Xnonmass, sX), axis=0)
    comby = np.concatenate( (ynonmass, sy2+sy))
    
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=5, early_exaggeration=4, learning_rate=10, 
                         init='pca', random_state=0, verbose=2, method='exact', n_iter=2000)
    t0 = time()
    X_tsne = tsne.fit_transform(combX)
    
    plot_embedding(combX, comby, combdata,
                   title="t-SNE embedding for D=2 for a subsample of normals done in (time %.2fs)" % (time() - t0),
                   plotextra=False)
                   
    ####################
    # MultinomialNB
    ####################       
    from sklearn.naive_bayes import MultinomialNB
    
    classifier = MultinomialNB()
    targets = dataSERcounts['class'].values
    npSERcounts = np.asarray(SERcounts)
    classifier.fit(npSERcounts, targets)
    
    # predict first 5 cases
    print targets[:5]
    predictions = classifier.predict(npSERcounts[:5,:])
    print predictions
    predictions_proba = classifier.predict_proba(npSERcounts[:5,:])
    print classifier.get_params()
    
    ####################
    # MANIFOLD LEARNING
    #################### 
    from time import time
    import matplotlib.pyplot as plt
    from matplotlib import offsetbox
    from matplotlib.offsetbox import TextArea, AnnotationBbox
    from sklearn import (manifold, datasets, decomposition, ensemble,
                         discriminant_analysis, random_projection)

    from sklearn.manifold import TSNE
    
    X = np.asarray(normSERcounts)
    y =  datanormSERcounts['class'].values
    y2 = datanormSERcounts['type'].values
    n_neighbors = 10
    
    # subset only nonmasses
    Xnonmass = X[datanormSERcounts['type'].values=='nonmass',:]
    ynonmass =  datanormSERcounts['class'].values[datanormSERcounts['type'].values=='nonmass']
    datanonmass = datanormSERcounts[datanormSERcounts['type'].values=='nonmass']
    print datanonmass.describe()
    
    #----------------------------------------------------------------------                  
    # Random 2D projection using a random unitary matrix
    print("Computing random projection")
    rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
    X_projected = rp.fit_transform(X)
    # plot 
    plot_embedding(X_projected, y, datanormSERcounts, title="Random Projection ",plotextra=False)

    #----------------------------------------------------------------------
    # Projection on to the first 2 principal components
    print("Computing PCA projection")
    t0 = time()
    X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
    plot_embedding(X_pca, y, datanormSERcounts,  
               title="Principal Components projection (time %.2fs)" % (time() - t0),
                plotextra=False)
                       
    #----------------------------------------------------------------------
    # Isomap projection 
    print("Computing Isomap embedding")
    t0 = time()
    X_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(X)
    plot_embedding(X_iso, y, datanormSERcounts,
                   title="Isomap projection (time %.2fs)" % (time() - t0),
                   plotextra=False)
                   
    print dataSERcounts

    #----------------------------------------------------------------------
    # Locally linear embedding 
    print("Computing LLE embedding")
    clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                          method='standard')
    t0 = time()
    X_lle = clf.fit_transform(X)
    print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
    plot_embedding(X_lle, y, datanormSERcounts,
                   title="Locally Linear Embedding (time %.2fs)" % (time() - t0),
                   plotextra=False)
                   
    #----------------------------------------------------------------------
    # Modified Locally linear embedding 
    print("Computing modified LLE embedding")
    clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                          method='modified')
    t0 = time()
    X_mlle = clf.fit_transform(X)
    print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
    plot_embedding(X_mlle, y, datanormSERcounts,
                   title="Modified Locally Linear Embedding (time %.2fs)" % (time() - t0),
                   plotextra=False)


    #----------------------------------------------------------------------
    # MDS  embedding 
    print("Computing MDS embedding")
    clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
    t0 = time()
    X_mds = clf.fit_transform(X)
    print("Done. Stress: %f" % clf.stress_)

    plot_embedding(X_mds, y, datanormSERcounts,
                   title="MDS embedding (time %.2fs)" % (time() - t0),
                   plotextra=False)
                   

    #----------------------------------------------------------------------
    # t-SNE embedding of the digits dataset
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    X_tsne = tsne.fit_transform(X)
    
    plot_embedding(X_tsne, y, datanormSERcounts,
                   title="t-SNE embedding of the digits (time %.2fs)" % (time() - t0),
                   plotextra=False)
                   
    """
                   