# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 12:58:03 2016

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


def run_nxG_extractfeature(nxGdatafeatures, featurename, normalizedflag=True, plotFeatflag=True, plotname=None):
    # define and extract
    extractedfeature = []
    
    allnxG_f = nxGdatafeatures[featurename].values
    # find min and max to discretize
    minmax_f_d = np.asarray([(min(f.values()), max(f.values())) for f in allnxG_f])
    minmax_f = [np.min(minmax_f_d, axis=0)[0], np.max(minmax_f_d, axis=0)[1]]
    # first discretize values
    discrvals_f= np.linspace(minmax_f[0], minmax_f[1], 11, endpoint=True)
    discrallnxG_f = [np.histogram(nxG_f.values(), bins=discrvals_f) for nxG_f in allnxG_f]
    discrallnxG_fcounts = [c  for (c,b) in discrallnxG_f]
    
    if(normalizedflag):
        # normalize by number of edges
        normdiscrallnxG_f = np.asarray([f.astype(np.float32)/np.sum(f) for f in discrallnxG_fcounts])

    if(plotFeatflag and plotname):  
        # as illustration plot the first 36 lesion features, label by gt
        fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(16, 10)); ax = axes.flat         
        for j in range(36):           
            ax[j].plot( normdiscrallnxG_f[j] )
            ax[j].set_xlabel(nxGdatafeatures['type'][j]+nxGdatafeatures['class'][j])
        plt.tight_layout()
        fig.savefig( plotname )
        
    if(normalizedflag):
        extractedfeature = normdiscrallnxG_f
    else:
        extractedfeature = discrallnxG_fcounts

    return extractedfeature
    
    
def run_nxG_extractfeature_categ(nxGdatafeatures, featurename, normalizedflag=True, plotFeatflag=True, plotname=None):
    # define and extract
    extractedfeature = []
    
    allnxG_f = nxGdatafeatures[featurename].values
    # find min and max to discretize
    minmax_f_d = np.asarray([(min(f.keys()), max(f.keys())) for f in allnxG_f])
    minmax_f = [np.min(minmax_f_d, axis=0)[0], np.max(minmax_f_d, axis=0)[1]]
    # first discretize values
    discrvals_f= np.linspace(minmax_f[0], minmax_f[1], minmax_f[1], endpoint=True)
    if(minmax_f[0]==0):
        discrvals_f= np.linspace(minmax_f[0], minmax_f[1], minmax_f[1]+1, endpoint=True)
    discrallnxG_f = []
    
    for anxG in allnxG_f:
        discr_f = {}
        for l,kn in enumerate(discrvals_f):
            if(kn in anxG.keys()):
                discr_f[kn] = [v for k,v in anxG.items() if k == kn][0]
            else:
                discr_f[kn] = 0.0
        # append
        discrallnxG_f.append( discr_f )
    
    if(plotFeatflag and plotname):  
        # as illustration plot the first 36 lesion features, label by gt
        fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(16, 10)); ax = axes.flat      
        sns.set_style("darkgrid", {"legend.frameon": True})
        for j in range(36):           
            sns.barplot(x=discrallnxG_f[j].keys(), y=discrallnxG_f[j].values(), ax=ax[j]); 
            ax[j].set_xlabel(nxGdatafeatures['type'][j]+nxGdatafeatures['class'][j])
        plt.tight_layout()
        fig.savefig( plotname )
        
    if(normalizedflag):
        extractedfeature = np.asarray([item.values() for item in discrallnxG_f])
    else:
        extractedfeature =  np.asarray([item.values() for item in discrallnxG_f])

    return extractedfeature    
    

def run_nxG_algorithms(saveFigs=True, returnnormFeatures=True, normalizedflag=True, plotFeatflag=True):
    # Get Root folder ( the directory of the script being run)
    path_rootFolder = os.path.dirname(os.path.abspath(__file__))
    os.chdir(path_rootFolder) 
    
    graphs_path = 'Z:\\Cristina\\Section3\\breast_MR_pipeline\\graphs'
    nxGfeatures_path = 'Z:\\Cristina\\Section3\\breast_MR_pipeline\\lesion_nxGfeatures'
    
    if not os.path.exists(nxGfeatures_path):
        os.mkdir(nxGfeatures_path)
    
    ########################################
    file_ids = open("listofprobmaps.txt","r") 
    file_ids.seek(0)
    line = file_ids.readline()
  
    nxGdatafeatures = pd.DataFrame({})
    nxGnormfeatures = []
    
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
            ###### 3) Examine nxG features according to graph properties (for more info and eg see http://localhost:8892/notebooks/mining_graphPatterns.ipynb)
            ######  Distance Measures (helpul for normalization)
            ######  a) nx.eccentricity
            ######        Return type: dictionary of eccentricity values keyed by node = eccentricity(G, v=None, sp=None)
            ######        [https://networkx.readthedocs.io/en/stable/reference/algorithms.distance_measures.html]
            ######        The eccentricity of a node v is the maximum distance from v to all other nodes in G.
            ###### Centrality
            ###### https://networkx.readthedocs.io/en/stable/reference/algorithms.centrality.html
            ######  b) centrality.degree_centrality(G)	Compute the degree centrality for nodes.
            ######  c) centrality.closeness_centrality(G)   Closeness centrality for nodes.
            ######  d) centrality.betweenness_centrality(G)   Compute the shortest-path betweenness centrality for nodes.
            ######  e) nx.katz_centrality(G) Compute the Katz centrality for the nodes of the graph G.
            ######
            ######  Clustering
            ######  http://networkx.readthedocs.io/en/stable/reference/algorithms.clustering.html
            ######  f) nx.clustering(G)	Compute the weighted clustering as the geometric average of the subgraph edge weights
            ######
            ######  Average degree connectivity
            ######  https://networkx.readthedocs.io/en/stable/reference/generated/networkx.algorithms.assortativity.average_degree_connectivity.html
            ######  g) nx.average_degree_connectivity(G[, source, ...])	Compute the average degree connectivity of graph.
            ######  h) nx.average_neighbor_degree(G[, source, target, ...])	Returns the average degree of the neighborhood of each node.
            ######
            ######  Rich Club
            ######  http://networkx.readthedocs.io/en/stable/reference/algorithms.rich_club.html
            ######  i) nx.rich_club_coefficient(G)	Compute the weighted clustering as the geometric average of the subgraph edge weights
            ############################# 
            nxG_ecc = nx.eccentricity(lesion_MST)
            nxG_centrality = centrality.degree_centrality(lesion_MST)
            nxG_closeness = centrality.closeness_centrality(lesion_MST)
            nxG_betweenness = centrality.betweenness_centrality(lesion_MST)
            nxG_katz_centrality = nx.katz_centrality(lesion_MST)
            nxG_clustering = nx.clustering(lesion_DEL, weight='weigth')
            nxG_average_degconn = nx.average_degree_connectivity(lesion_MST)  
            nxG_average_neighbor_degree = nx.average_neighbor_degree(lesion_MST)
            nxG_rich_club = nx.rich_club_coefficient(lesion_MST)
     
             # plot
            fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 10))
            ax = axes.flat
            sns.set_style("darkgrid", {"legend.frameon": True})
            # format
            pd_nxG_ecc = pd.Series(nxG_ecc.values(), name="Eccentricity")
            pd_nxG_centrality = pd.Series(nxG_centrality.values(), name="Centrality")
            pd_nxG_closeness = pd.Series(nxG_closeness.values(), name="Closeness")
            pd_nxG_betweenness = pd.Series(nxG_betweenness.values(), name="Betweenness")
            pd_nxG_katz_centrality = pd.Series(nxG_katz_centrality.values(), name="Katz centrality")
            pd_nxG_clustering = pd.Series(nxG_clustering.values(), name="Clustering")
            pd_nxG_average_degconn = pd.Series(nxG_average_degconn.values(), name="average Degree Connectivity")
            pd_nxG_average_neighdeg = pd.Series(nxG_average_neighbor_degree.values(), name="average Neighbor Degree")
            pd_nxG_rich_club = pd.Series(nxG_rich_club.values(), name="Rich club")
            # plot
            sns.distplot(pd_nxG_ecc, label="Eccentricity", ax=ax[0])
            sns.distplot(pd_nxG_centrality, label="Centrality", ax=ax[1])
            sns.distplot(pd_nxG_closeness, label="Closeness", ax=ax[2])
            sns.distplot(pd_nxG_betweenness, label="Betweenness", ax=ax[3])
            sns.distplot(pd_nxG_katz_centrality, label="Katz centrality", ax=ax[4])
            sns.distplot(pd_nxG_clustering, label="Clustering", ax=ax[5])
            sns.barplot(x=nxG_average_degconn.keys(), y=nxG_average_degconn.values(), ax=ax[6]); ax[6].set(xlabel='average Degree Connectivity')
            sns.distplot(pd_nxG_average_neighdeg, label="average Neighbor Degree", ax=ax[7])
            sns.barplot(x=nxG_rich_club.keys(), y=nxG_rich_club.values(), ax=ax[8]); ax[8].set(xlabel='Rich club')
            labelLesion = 'lesion_id_{}_{}_{}_nxGfeatures_{}{}_{}'.format(str(lesion_id),fStudyID.zfill(4),AccessionN,cond,BenignNMaligNAnt,Diagnosis)
            plt.suptitle(labelLesion)
             
            #show and save  
            if(saveFigs):
                fig.savefig( os.path.join( nxGfeatures_path,labelLesion+'.pdf' ) )
                
            #############################
            ###### 4) append to graphvector of nxGfeatures
            #############################
            # to build dataframe    
            rows = []
            index = []     
            rows.append({'nxG_ecc': nxG_ecc, 
                         'nxG_centrality': nxG_centrality,
                         'nxG_closeness': nxG_closeness,
                         'nxG_betweenness': nxG_betweenness,
                         'nxG_katz_centrality': nxG_katz_centrality,
                         'nxG_clustering': nxG_clustering,
                         'nxG_average_degconn': nxG_average_degconn,
                         'nxG_average_neighbor_degree': nxG_average_neighbor_degree,
                         'nxG_rich_club': nxG_rich_club,
                         'class': BenignNMaligNAnt, 
                         'type': cond})         
            index.append(labelLesion)
            
            # append counts to master lists
            nxGdatafeatures = nxGdatafeatures.append( pd.DataFrame(rows, index=index) )
           
            ## next line
            line = file_ids.readline()  
            plt.close()
     
        except:
            continue
            
    if(returnnormFeatures):
        ########################
        #### 1) nxG_eccentricity
        ########################
        normdiscrallnxG_ecc = run_nxG_extractfeature(nxGdatafeatures, 'nxG_ecc', normalizedflag=True, plotFeatflag=True, 
                                                       plotname=os.path.join(nxGfeatures_path,'nxG_eccentricity_discretized_10binsize.pdf') )
        print "extracted normdiscrallnxG_ecc with %i cases and %i total features" % (normdiscrallnxG_ecc.shape[0], normdiscrallnxG_ecc.shape[1])                                                                                                              
              
        ########################
        #### 2) nxG_centrality
        ########################
        normdiscrallnxG_centr = run_nxG_extractfeature(nxGdatafeatures, 'nxG_centrality', normalizedflag=True, plotFeatflag=True, 
                                                       plotname=os.path.join(nxGfeatures_path,'nxG_centrality_discretized_10binsize.pdf') )
        print "extracted normdiscrallnxG_centr with %i cases and %i total features" % (normdiscrallnxG_centr.shape[0], normdiscrallnxG_centr.shape[1])                                                                                                              
           
        ########################
        #### 3) nxG_closeness
        ########################
        normdiscrallnxG_closen = run_nxG_extractfeature(nxGdatafeatures, 'nxG_closeness', normalizedflag=True, plotFeatflag=True, 
                                                       plotname=os.path.join(nxGfeatures_path,'nxG_closeness_discretized_10binsize.pdf') )
        print "extracted normdiscrallnxG_closen with %i cases and %i total features" % (normdiscrallnxG_closen.shape[0], normdiscrallnxG_closen.shape[1])                                                                                                              
                        
        ########################
        #### 4) nxG_betweenness
        ########################
        normdiscrallnxG_betwen = run_nxG_extractfeature(nxGdatafeatures, 'nxG_betweenness', normalizedflag=True, plotFeatflag=True, 
                                                       plotname=os.path.join(nxGfeatures_path,'nxG_betweenness_discretized_10binsize.pdf') )
        print "extracted normdiscrallnxG_betwen with %i cases and %i total features" % (normdiscrallnxG_betwen.shape[0], normdiscrallnxG_betwen.shape[1])                                                                                                              
        
        ########################
        #### 5) nxG_katz_centrality
        ########################
        normdiscrallnxG_katz = run_nxG_extractfeature(nxGdatafeatures, 'nxG_katz_centrality', normalizedflag=True, plotFeatflag=True, 
                                                       plotname=os.path.join(nxGfeatures_path,'nxG_katz_centrality_discretized_10binsize.pdf') )
        print "extracted normdiscrallnxG_katz with %i cases and %i total features" % (normdiscrallnxG_katz.shape[0], normdiscrallnxG_katz.shape[1])                                                                                                                     
        
        ########################
        #### 6) nxG_clustering
        ########################
        normdiscrallnxG_cluster = run_nxG_extractfeature(nxGdatafeatures, 'nxG_clustering', normalizedflag=True, plotFeatflag=True, 
                                                       plotname=os.path.join(nxGfeatures_path,'nxG_clustering_discretized_10binsize.pdf') )
        print "extracted normdiscrallnxG_cluster with %i cases and %i total features" % (normdiscrallnxG_cluster.shape[0], normdiscrallnxG_cluster.shape[1])                                                                                                                     
         
        ########################
        #### 7) nxG_average_degconn
        ########################
        normdiscrallnxG_degconn = run_nxG_extractfeature_categ(nxGdatafeatures, 'nxG_average_degconn', normalizedflag=True, plotFeatflag=True, 
                                                       plotname=os.path.join(nxGfeatures_path,'nxG_average_degconn_discretized_10binsize.pdf') )
        print "extracted normdiscrallnxG_degconn with %i cases and %i total features" % (normdiscrallnxG_degconn.shape[0], normdiscrallnxG_degconn.shape[1])                                                                                                                               
        
        ########################
        #### 8) nxG_average_neighbor_degree
        ########################
        normdiscrallnxG_averNdegree = run_nxG_extractfeature(nxGdatafeatures, 'nxG_average_neighbor_degree', normalizedflag=True, plotFeatflag=True, 
                                                       plotname=os.path.join(nxGfeatures_path,'nxG_average_neighbor_degree_discretized_10binsize.pdf') )
        print "extracted normdiscrallnxG_averNdegree with %i cases and %i total features" % (normdiscrallnxG_averNdegree.shape[0], normdiscrallnxG_averNdegree.shape[1])                                                                                                                     
         
        ########################
        #### 9) nxG_rich_club
        ########################
        normdiscrallnxG_richC = run_nxG_extractfeature_categ(nxGdatafeatures, 'nxG_rich_club', normalizedflag=True, plotFeatflag=True, 
                                                       plotname=os.path.join(nxGfeatures_path,'nxG_rich_club_discretized_10binsize.pdf') )
        print "extracted normdiscrallnxG_richC with %i cases and %i total features" % (normdiscrallnxG_richC.shape[0], normdiscrallnxG_richC.shape[1])                                                       
        
        ## Finally: concatenates along the 1st axis (rows of the 1st, then rows of the 2nd)       
        if(normalizedflag):            
            nxGnormfeatures = np.concatenate((normdiscrallnxG_ecc, normdiscrallnxG_centr, normdiscrallnxG_closen, normdiscrallnxG_betwen,
                                              normdiscrallnxG_katz, normdiscrallnxG_cluster,
                                              normdiscrallnxG_degconn, normdiscrallnxG_averNdegree, normdiscrallnxG_richC), axis=1)
            print "returning nxGnormfeatures consisting on %i cases and %i total features" % (nxGnormfeatures.shape[0], nxGnormfeatures.shape[1])                                              
            
        
    return nxGdatafeatures, nxGnormfeatures  
    
if __name__ == '__main__':
    """
    usage:
    
    from nxG_algorithms import *
    from nxG_visualize import *

    
    ########################################
    ## to extract nxG features
    ########################################
    nxGdatafeatures, nxGnormfeatures = run_nxG_algorithms(saveFigs=True, returnnormFeatures=True, normalizedflag=True, plotFeatflag=True)
    
    # save all nxGdatafeatures
    nxGfeatures_path = 'Z:\\Cristina\\Section3\\breast_MR_pipeline\\lesion_nxGfeatures'
    nxGdatafeatures_allLesions = gzip.open(os.path.join(nxGfeatures_path,'nxGdatafeatures_allLesions_10binsize.pklz'), 'wb')
    pickle.dump(nxGdatafeatures, nxGdatafeatures_allLesions, protocol=pickle.HIGHEST_PROTOCOL)
    nxGdatafeatures_allLesions.close()
    
    nxGnormfeatures_allLesions = gzip.open(os.path.join(nxGfeatures_path,'nxGnormfeatures_allLesions_10binsize.pklz'), 'wb')
    pickle.dump(nxGnormfeatures, nxGnormfeatures_allLesions, protocol=pickle.HIGHEST_PROTOCOL)
    nxGnormfeatures_allLesions.close()       
    
    ########################################
    # to load all nxGdatafeatures
    nxGfeatures_path = 'Z:\\Cristina\\Section3\\breast_MR_pipeline\\lesion_nxGfeatures'
    with gzip.open(os.path.join(nxGfeatures_path,'nxGdatafeatures_allLesions_10binsize.pklz'), 'rb') as fu:
        nxGdatafeatures = pickle.load(fu)
    
    with gzip.open(os.path.join(nxGfeatures_path,'nxGnormfeatures_allLesions_10binsize.pklz'), 'rb') as fu:
        nxGnormfeatures = pickle.load(fu)
        
    # set up some parameters and define labels
    X = nxGnormfeatures
    print"Input data to t-SNE is mxn-dimensional with m = %i discretized SER bins" % X.shape[1]
    print"Input data to t-SNE is mxn-dimensional with n = %i cases" % X.shape[0]
    y =  nxGdatafeatures['class'].values
    y2 = nxGdatafeatures['type'].values
    
    tsne = TSNE(n_components=2, perplexity=9, early_exaggeration=7, learning_rate=320, 
                 init='pca', random_state=0, verbose=2, method='exact')      
    X_tsne = tsne.fit_transform(X)
    y_tsne = y #y2+y
    
    ## plot TSNE
    plot_embedding(X_tsne, y, nxGdatafeatures, title='nxGdatafeatures',  plotextra=False)
    
    ########################################
    ## adding SERedgew
    ########################################
    # to load SERw matrices for all lesions
    SER_edgesw_path = 'Z:\\Cristina\\Section3\\breast_MR_pipeline\\lesion_SER_edgesw'
    with gzip.open(os.path.join(SER_edgesw_path,'allLesions_10binsizenormSERcounts_probC_x.pklz'), 'rb') as fu:
        normSERcounts = pickle.load(fu)
    
    with gzip.open(os.path.join(SER_edgesw_path,'allLesions_10binsizedatanormSERcounts_probC_x.pklz'), 'rb') as fu:
        datanormSERcounts = pickle.load(fu)
        
    combX = np.concatenate((nxGnormfeatures, np.asarray(normSERcounts)), axis=1)
     
    tsne = TSNE(n_components=2, perplexity=9, early_exaggeration=7, learning_rate=320, 
                 init='pca', random_state=0, verbose=2, method='exact')      
    X_tsne = tsne.fit_transform(combX)
    y_tsne = y #y2+y
    
    ## plot TSNE
    plot_embedding(X_tsne, y, nxGdatafeatures, title='nxGdatafeatures',  plotextra=False)
    
    
    """