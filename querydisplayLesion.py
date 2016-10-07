# -*- coding: utf-8 -*-
"""
Created on Tue Feb 09 10:35:58 2016

@ author (C) Cristina Gallego, University of Toronto
"""

import sys, os
import string
import datetime
from numpy import *
import pandas as pd

from functionsData import *
from newFeatures import *

if __name__ == '__main__':    
    # Get Root folder ( the directory of the script being run)
    path_rootFolder = os.path.dirname(os.path.abspath(__file__))
    print path_rootFolder
       
    # Open filename list
    print sys.argv[1]
    file_ids = open(sys.argv[1],"r") #file_ids = open("liststage1T2updatedFeatures.txt","r")
    file_ids.seek(0)
    line = file_ids.readline()
    l=637
       
    while ( line ) : 
        # Get the line: id  	id	CAD study #	Accession #	Date of Exam	Pathology Diagnosis 
        print(line)
        fileline = line.split()
        lesion_id = int(fileline[0] )
        newlesion_id = l
        StudyID = fileline[1] 
        AccessionN = fileline[2]  
        dateID = fileline[3]
        sideB = fileline[4]
        Diagnosis = fileline[5:]
        
         # Format query StudyID
        if (len(StudyID) >= 4 ): fStudyID=StudyID
        if (len(StudyID) == 3 ): fStudyID='0'+StudyID
        if (len(StudyID) == 2 ): fStudyID='00'+StudyID
        if (len(StudyID) == 1 ): fStudyID='000'+StudyID
  
        #############################
        ## Load helper functions
        funcD = functionsData()
        
        #############################
        ###### 1) Querying Research database for clinical, pathology, radiology data
        #############################
        [img_folder, MorNM, BenignNMaligNAnt, Diagnosis, casesFrame, MorNMcase, T2info] = funcD.querylocalDatabase_wRad(lesion_id)        

        AccessionN = casesFrame['exam_a_number_txt']
        DicomExamNumber = casesFrame['exam_img_dicom_txt']
        ## for old DicomExamNumber         
        #AccessionN = DicomExamNumber
        dateID = casesFrame['exam_dt_datetime']
        finding_side = casesFrame['exam_find_side_int']
        if(finding_side=='L'):
            finding_side='Left'
        if(finding_side=='R'):
            finding_side='Right'

        pathSegment = 'C:\Users\windows\Documents'+os.sep+'repoCode-local'+os.sep+'registerT2andfeatures'+os.sep+'segmentations'
        nameSegment = casesFrame['lesionfile'] 
        lesion_label = casesFrame['lesion_label']
        DynSeries_id = MorNMcase['DynSeries_id']
        T2Series_id = MorNMcase['T2Series_id']
        
        #############################                  
        ###### 3) Query biomatrix for rad report
        ############################# 
        ## when querying biomatrix
        #radioinfo = funcD.queryRadioData(fStudyID, dateID)
        # or query local
        radioinfo = funcD.querylocalRadioData(lesion_id)        
        radioinfo['exam_dt_datetime'] = casesFrame['exam_dt_datetime']
    
        #############################                  
        ###### 3) Visualize and load, check with annotations
        #############################       
        print "\n Preload volumes and segmentation..."
        [series_path, phases_series, annotationsfound] = funcD.preloadSegment(img_folder, lesion_id, StudyID, AccessionN, DicomExamNumber, DynSeries_id, T2Series_id, pathSegment, nameSegment)
        
        print "\n Visualize and load DCE lesion..."
        newpathSegment = 'C:\Users\windows\Documents'+os.sep+'repoCode-local'+os.sep+'querydisplayLesion'+os.sep+'outsegmentations'
        newnameSegment = str(l)+'_'+fStudyID+'_'+AccessionN+'.vtk'
        funcD.loadSegment(newpathSegment, newnameSegment)    
        
        ############################# IF REPEATING SEGMENTATION
        print "\n Resegment or not"
        resegorNot=0
        resegorNot = int(raw_input('type 1 to Resegment: '))
        if resegorNot == 1:
            LesionZslice = funcD.loadDisplay.zImagePlaneWidget.GetSliceIndex()
            funcD.segmentLesion(LesionZslice, newnameSegment)
        
        #############################
        ###### 4) Extract Dynamic features
        #############################
        [dyn_inside, dyn_contour] = funcD.extract_dyn(series_path, phases_series, funcD.lesion3D)
        
        #############################
        ###### 5) Extract Morphology features
        ############################# 
        morphofeatures = funcD.extract_morph(series_path, phases_series, funcD.lesion3D)
        
        #############################        
        ###### 6) Extract Texture features
        #############################
        patchdirname = 'C:\Users\windows\Documents'+os.sep+'repoCode-local'+os.sep+'querydisplayLesion'+os.sep+'patchdir'+os.sep+str(l)+'_'+fStudyID+'_'+AccessionN
        texturefeatures = funcD.extract_text(series_path, phases_series, funcD.lesion3D, patchdirname)       
        
        #############################
        # 4) Extract Lesion and Muscle Major pectoralies signal                                   
        ############################# 
        line_muscleVOI = T2info['bounds_muscleSI']
        line_muscleVOI = line_muscleVOI.rstrip()
        lmuscle = line_muscleVOI[line_muscleVOI.find('[')+1:line_muscleVOI.find(']')].split(",")
        bounds_muscleSI = [float(lmuscle[0]), float(lmuscle[1]), float(lmuscle[2]), float(lmuscle[3]), float(lmuscle[4]), float(lmuscle[5]) ]
        print "\n bounds_muscleSI from file:"
        print bounds_muscleSI
        
        #############################
        # 5) Extract T2 features                            
        #############################
        [T2_muscleSI, muscle_scalar_range, bounds_muscleSI, 
         T2_lesionSI, lesion_scalar_range, LMSIR, morphoT2features, textureT2features] = funcD.T2_extract(T2Series_id, funcD.lesion3D, bounds_muscleSI, patchdirname)    
        
        #############################
        ###### 8) Extract new features from each DCE-T1 and from T2 using segmented lesion
        #############################
        newfeatures = newFeatures(funcD.load, funcD.loadDisplay)
        [deltaS, t_delta, centerijk] = newfeatures.extract_MRIsamp(series_path, phases_series, funcD.lesion3D, T2Series_id)
        
        # generate nodes from segmantation 
        [nnodes, curveT, earlySE, dce2SE, dce3SE, lateSE, ave_T2, prop] = newfeatures.generateNodesfromKmeans(deltaS['i0'], deltaS['j0'], deltaS['k0'], deltaS, centerijk, T2Series_id)    
        [kmeansnodes, d_euclideanNodes] = prop
        
        # pass nodes to lesion graph
        G = newfeatures.createGraph(nnodes, curveT, prop)                   
    
        [degreeC, closenessC, betweennessC, no_triangles, no_con_comp] = newfeatures.analyzeGraph(G)        
        network_measures = [degreeC, closenessC, betweennessC, no_triangles, no_con_comp]
        
        #############################
        ###### 10) End and Send record to DB
        #############################
        DynSeries_id = MorNMcase['DynSeries_id']
        T2Series_id = MorNMcase['T2Series_id']
       
        funcD.addRecordDB_lesion(newnameSegment, fStudyID, DicomExamNumber, dateID, casesFrame, finding_side, MorNMcase, MorNM, Diagnosis, newlesion_id, BenignNMaligNAnt,  DynSeries_id, T2Series_id)
                           
        funcD.addRecordDB_features(newlesion_id, dyn_inside, dyn_contour, morphofeatures, texturefeatures)   
        
        funcD.addRecordDB_T2(newlesion_id, T2Series_id, T2info, morphoT2features, textureT2features, T2_muscleSI, muscle_scalar_range, bounds_muscleSI, T2_lesionSI, lesion_scalar_range, LMSIR)
    
        #############################
        print "\n Adding record case to stage1"
        funcD.addRecordDB_stage1(newlesion_id, d_euclideanNodes, earlySE, dce2SE, dce3SE, lateSE, ave_T2, network_measures)

        funcD.addRecordDB_radiology(newlesion_id, radioinfo)
        
        pylab.close('all') 
  
        #############################
        ###### 10) End and Send record to DB
        #############################
#        print "\n Delete or not"
#        DelorNot = int(raw_input('type 1 to Delete: '))
#        if DelorNot == 1:
#            funcD.querylocal.deleteby_lesionid(lesion_id)

        #############################
        ## continue to next case
        line = file_ids.readline()
        print line
        l=l+1
       
    file_ids.close()
