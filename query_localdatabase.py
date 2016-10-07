# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 16:48:07 2014

@ author (C) Cristina Gallego, University of Toronto
"""
import sys, os
import string
import datetime
from numpy import *
import pandas as pd

import mylocaldatabase
from mylocalbase import localengine

from sqlalchemy.orm import sessionmaker

#!/usr/bin/env python
class Querylocal(object):
    """
    USAGE:
    =============
    localdata = Querylocal()
    """

    def __init__(self):
        """ initialize QueryDatabase """


    def queryby_lesionid(self, lesion_id):
        """
        run : Query by Lesion_id on local database

        Inputs
        ======
        lesion_id : (int)    My CADStudy lesion_id

        Output
        lesion record
        ======
        """
        # Create the database: the Session.
        self.Session = sessionmaker()
        self.Session.configure(bind=localengine)  # once engine is available
        session = self.Session() #instantiate a Session

        # for cad_case in session.query(Cad_record).order_by(Cad_record.pt_id):
        #     print cad_case.pt_id, cad_case.cad_pt_no_txt, cad_case.latest_mutation_status_int
        for lesion in session.query(mylocaldatabase.Lesion_record,mylocaldatabase.Radiology_record).\
            filter(mylocaldatabase.Radiology_record.lesion_id==mylocaldatabase.Lesion_record.lesion_id).\
            filter(mylocaldatabase.Lesion_record.lesion_id == str(lesion_id)):
            # print results
            if not lesion:
                print "lesion is empty"

        return lesion
        
    def deleteby_lesionid(self, lesion_id):
        """
        run : Delete whole records by Lesion_id on local database

        Inputs
        ======
        lesion_id : (int)    My CADStudy lesion_id

        Output
        lesion record
        ======
        """
        # Create the database: the Session.
        self.Session = sessionmaker()
        self.Session.configure(bind=localengine)  # once engine is available
        session = self.Session() #instantiate a Session

        ## delete with query (supported one table at a time)
        session.query(mylocaldatabase.Lesion_record).\
        filter(mylocaldatabase.Lesion_record.lesion_id == str(lesion_id)).\
             delete(synchronize_session='evaluate')
             
        session.query(mylocaldatabase.Radiology_record).\
        filter(mylocaldatabase.Radiology_record.lesion_id == lesion_id).\
             delete(synchronize_session=False)             
             
        session.query(mylocaldatabase.Segment_record).\
        filter(mylocaldatabase.Segment_record.lesion_id == lesion_id).\
             delete(synchronize_session=False) 

        session.query(mylocaldatabase.Stage1_record).\
        filter(mylocaldatabase.Stage1_record.lesion_id == lesion_id).\
             delete(synchronize_session=False) 

        session.query(mylocaldatabase.Mass_record).\
        filter(mylocaldatabase.Mass_record.lesion_id == lesion_id).\
             delete(synchronize_session=False) 

        session.query(mylocaldatabase.Nonmass_record).\
        filter(mylocaldatabase.Nonmass_record.lesion_id == lesion_id).\
             delete(synchronize_session=False) 

        session.query(mylocaldatabase.Foci_record).\
        filter(mylocaldatabase.Foci_record.lesion_id == lesion_id).\
             delete(synchronize_session=False) 
             
        session.query(mylocaldatabase.Dynamic_features).\
        filter(mylocaldatabase.Dynamic_features.lesion_id == lesion_id).\
             delete(synchronize_session=False)      
             
        session.query(mylocaldatabase.Morpho_features).\
        filter(mylocaldatabase.Morpho_features.lesion_id == lesion_id).\
             delete(synchronize_session=False)

        session.query(mylocaldatabase.Texture_features).\
        filter(mylocaldatabase.Texture_features.lesion_id == lesion_id).\
             delete(synchronize_session=False)             
             
        session.query(mylocaldatabase.T2_features).\
        filter(mylocaldatabase.T2_features.lesion_id == lesion_id).\
             delete(synchronize_session=False)             
             
        print "Deleting lesion records...."
        
        # Finally send records to database
        try:
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()

        return 



    def query_withT2(self, lesion_id):

         # Create the database: the Session.
        self.Session = sessionmaker()
        self.Session.configure(bind=localengine)  # once engine is available
        session = self.Session() #instantiate a Session

        # for cad_case in session.query(Cad_record).order_by(Cad_record.pt_id):
        #     print cad_case.pt_id, cad_case.cad_pt_no_txt, cad_case.latest_mutation_status_int
        T2record = []
        for T2record in session.query(mylocaldatabase.T2_features).\
            filter(mylocaldatabase.T2_features.lesion_id == str(lesion_id)):
            # print results
            if not T2record:
                print "lesion T2record is empty"


        return T2record


    def queryBiomatrixBIRADSage(self, session, fStudyID, redateID):
        """
        run : Query by StudyID/AccesionN pair study to local folder NO GRAPICAL INTERFACE. default print to output

        Inputs
        ======
        fStudyID : (int)    CAD fStudyID
        redateID : (int)  CAD StudyID Data of exam (format yyyy-mm-dd)

        Output
        ======
        """
        datainfo = [];
        dfBIRADSfinding = pd.DataFrame()

        for pt, cad, exam, finding in session.query(database.Pt_record, database.Cad_record, database.Exam_record, database.Exam_Finding).\
                     filter(database.Pt_record.pt_id==database.Cad_record.pt_id).\
                     filter(database.Cad_record.pt_id==database.Exam_record.pt_id).\
                     filter(database.Exam_record.pt_exam_id==database.Exam_Finding.pt_exam_id).\
                     filter(database.Cad_record.cad_pt_no_txt == str(fStudyID)).\
                     filter(database.Exam_record.exam_dt_datetime == str(redateID)).all():                      
                        
            # print results
            if not cad:
                print "cad is empty"
            if not exam:
                print "exam is empty"
            if not finding:
                print "finding is empty"
            
            datainfo.append( [cad.cad_pt_no_txt, pt.anony_dob_datetime, exam.a_number_txt, exam.exam_dt_datetime, exam.mri_cad_status_txt,
                                    exam.comment_txt,
                                    exam.original_report_txt,
                                    finding.side_int, finding.all_birads_scr_int,
                                    finding.mri_mass_yn, finding.mri_nonmass_yn, finding.mri_foci_yn] )
            
            ################### write output query to pandas frame.
            colLabels = ("cad.cad_pt_no_txt", "pt.anony_dob_datetime", "exam.a_number_txt", "exam.exam_dt_datetime", "exam.mri_cad_status_txt",
                     "exam.comment_txt", "exam.original_report_txt",
                     "finding.side_int", "finding.all_birads_scr_int",
                     "finding.mri_mass_yn", "finding.mri_nonmass_yn", "finding.mri_foci_yn")
            dinfo = pd.DataFrame(data=datainfo, columns=colLabels)  
            dfBIRADSfinding = dfBIRADSfinding.append(dinfo)                                  

        # write output query to pandas frame.
        print len(dfBIRADSfinding) 
        print(dfBIRADSfinding['exam.original_report_txt'].iloc[0])
        print(dfBIRADSfinding['finding.all_birads_scr_int'])

        return dfBIRADSfinding

