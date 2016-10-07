# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 16:09:01 2014

@author: Cristina Gallego
"""
import sys, os
import string
import datetime
import numpy as np

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship, backref

from mylocalbase import localengine

Base = declarative_base(localengine)

#  created a Cad_record mapping 
class Lesion_record(Base):
    """Base for Exam_record class using Declarative. for table lesion
    attributes:
        self.lesion_id = lesion_id
        self.cad_pt_no_txt = cad_id
        self.anony_dob_datetime = cad_dob
        self.exam_img_dicom_txt = dicom_no
        self.exam_a_number_txt = accession_no
        self.exam_dt_datetime = exam_date
        self.exam_mri_cad_status_txt = cad_status
        self.cad_latest_mutation_status_int = mutation
        self.exam_find_mri_mass_yn = mass_yn
        self.exam_find_mri_nonmass_yn = nonmass_yn
        self.exam_find_mri_foci_yn = foci_yn
        self.exam_find_side_int = finding_side
        self.proc_pt_procedure_id = proc_id
        self.proc_proc_dt_datetime = proc_date
        self.proc_proc_side_int = proc_side
        self.proc_proc_source_int = proc_source
        self.proc_proc_guid_int = proc_guid
        self.proc_proc_tp_int = proc_type
        self.proc_lesion_comments_txt = lesion_comments
        self.proc_original_report_txt = original_report
        self.find_curve_int = curve_int
        self.find_mri_dce_init_enh_int = dce_init
        self.find_mri_dce_delay_enh_int = dce_delay
        self.BIRADS = BIRADS
        self.lesion_label = label
        self.lesion_diagnosis = diagnosis    
    """
    __tablename__ = 'lesion'
    __table_args__ = {'autoload':True}
    lesion_id = Column(Integer, primary_key=True)
    mass_lesion = relationship("Mass_record", backref=backref('lesion', order_by=lesion_id))
    nonmass_lesion = relationship("Nonmass_record", backref=backref('lesion', order_by=lesion_id))
    foci_lesion = relationship("Foci_record", backref=backref('lesion', order_by=lesion_id))
    f_dynamic = relationship("Dynamic_features", backref=backref('lesion', order_by=lesion_id))
    f_morphology = relationship("Morpho_features", backref=backref('lesion', order_by=lesion_id))
    f_texture = relationship("Texture_features", backref=backref('lesion', order_by=lesion_id))
    f_T2 = relationship("T2_features", backref=backref('lesion', order_by=lesion_id))
    annotation = relationship("Annot_record", backref=backref('lesion', order_by=lesion_id))
    segmentation = relationship("Segment_record", backref=backref('lesion', order_by=lesion_id))
    radiology = relationship("Radiology_record", backref=backref('lesion', order_by=lesion_id))
        
    def __init__(self, lesionfile, cad_id, cad_dob, dicom_no, accession_no, exam_date, cad_status, mutation, mass_yn, nonmass_yn, foci_yn, finding_side, proc_id, proc_date, proc_side, proc_source, proc_guid, proc_type, lesion_comments, original_report, curve_int, dce_init, dce_delay, BIRADS, label, diagnosis):      
        self.lesionfile = lesionfile
        self.cad_pt_no_txt = cad_id
        self.anony_dob_datetime = cad_dob
        self.exam_img_dicom_txt = dicom_no
        self.exam_a_number_txt = accession_no
        self.exam_dt_datetime = exam_date
        self.exam_mri_cad_status_txt = cad_status
        self.cad_latest_mutation_status_int = mutation
        self.exam_find_mri_mass_yn = mass_yn
        self.exam_find_mri_nonmass_yn = nonmass_yn
        self.exam_find_mri_foci_yn = foci_yn
        self.exam_find_side_int = finding_side
        self.proc_pt_procedure_id = proc_id
        self.proc_proc_dt_datetime = proc_date
        self.proc_proc_side_int = proc_side
        self.proc_proc_source_int = proc_source
        self.proc_proc_guid_int = proc_guid
        self.proc_proc_tp_int = proc_type
        self.proc_lesion_comments_txt = lesion_comments
        self.proc_original_report_txt = original_report
        self.find_curve_int = curve_int
        self.find_mri_dce_init_enh_int = dce_init
        self.find_mri_dce_delay_enh_int = dce_delay
        self.BIRADS = BIRADS
        self.lesion_label = label
        self.lesion_diagnosis = diagnosis
        
    def __repr__(self):
        return "<Lesion_record(lesion_id='%s', cad_pt='%s', dicom='%s', a_number='%s', exam_date='%s')>" % (self.lesion_id, self.cad_pt_no_txt, self.exam_img_dicom_txt, self.exam_a_number_txt, self.exam_dt_datetime)


#  created a Mass_record mapping 
class Mass_record(Base):
    """Base for mass_lesion class using Declarative. for table mass_lesion
    attributes:
        self.lesion_id = lesion_id
        self.BenignNMaligNAnt = BenignNMaligNAnt
        self.DynSeries_id = DynSeries_id
        self.T2Series_id = T2Series_id
        self.find_mammo_n_mri_mass_shape_int = mri_mass_shape
        self.find_mri_mass_margin_int = mri_mass_margin
    """
    __tablename__ = 'mass_lesion'
    __table_args__ = {'autoload':True}
    mass_id = Column(Integer, primary_key=True)
    lesion_id = Column(Integer, ForeignKey('lesion.lesion_id'))
    
    def __init__(self, lesion_id, BenignNMaligNAnt, DynSeries_id, T2Series_id, mri_mass_shape, mri_mass_margin):
        self.lesion_id = lesion_id
        self.BenignNMaligNAnt = BenignNMaligNAnt
        self.DynSeries_id = DynSeries_id
        self.T2Series_id = T2Series_id
        self.find_mammo_n_mri_mass_shape_int = mri_mass_shape
        self.find_mri_mass_margin_int = mri_mass_margin
        
    def __repr__(self):
        return "<Mass_record(lesion_id='%s', BenignNMaligNAnt='%s', DynSeries_id='%s', T2Series_id='%s', mri_mass_shape='%s', , mri_mass_margin='%s')>" % (self.lesion_id, self.BenignNMaligNAnt, self.DynSeries_id, self.T2Series_id, self.find_mammo_n_mri_mass_shape_int, self.find_mri_mass_margin_int)


#  created a Mass_record mapping 
class Nonmass_record(Base):
    """Base for mass_lesion class using Declarative. for table mass_lesion
    attributes:
        self.lesion_id = lesion_id
        self.BenignNMaligNAnt = BenignNMaligNAnt
        self.DynSeries_id = DynSeries_id
        self.T2Series_id = T2Series_id
        self.find_mri_nonmass_dist_int = mri_nonmass_dist
        self.find_mri_nonmass_int_enh_int = mri_nonmass_int_enh   
    """
    __tablename__ = 'nonmass_lesion'
    __table_args__ = {'autoload':True}
    nonmass_id = Column(Integer, primary_key=True)
    lesion_id = Column(Integer, ForeignKey('lesion.lesion_id'))
        
    def __init__(self, lesion_id, BenignNMaligNAnt, DynSeries_id, T2Series_id, mri_nonmass_dist, mri_nonmass_int_enh):
        self.lesion_id = lesion_id
        self.BenignNMaligNAnt = BenignNMaligNAnt
        self.DynSeries_id = DynSeries_id
        self.T2Series_id = T2Series_id
        self.find_mri_nonmass_dist_int = mri_nonmass_dist
        self.find_mri_nonmass_int_enh_int = mri_nonmass_int_enh
        
    def __repr__(self):
        return "<Nonmass_record(lesion_id='%s', BenignNMaligNAnt='%s', DynSeries_id='%s', T2Series_id='%s', mri_nonmass_dist='%s', , mri_nonmass_int_enh='%s')>" % (self.lesion_id, self.BenignNMaligNAnt, self.DynSeries_id, self.T2Series_id, self.find_mri_nonmass_dist_int, self.find_mri_nonmass_int_enh_int)


#  created a Foci_record mapping 
class Foci_record(Base):
    """Base for Foci_record class using Declarative. for table foci_lesion
    attributes:
        self.lesion_id = lesion_id
        self.BenignNMaligNAnt = BenignNMaligNAnt
        self.DynSeries_id = DynSeries_id
        self.T2Series_id = T2Series_id
        self.mri_foci_distr_int = mri_foci_dist
    """
    __tablename__ = 'foci_lesion'
    __table_args__ = {'autoload':True}
    foci_id = Column(Integer, primary_key=True)
    lesion_id = Column(Integer, ForeignKey('lesion.lesion_id'))
        
    def __init__(self, lesion_id, BenignNMaligNAnt, DynSeries_id, T2Series_id, mri_foci_dist):
        self.lesion_id = lesion_id
        self.BenignNMaligNAnt = BenignNMaligNAnt
        self.DynSeries_id = DynSeries_id
        self.T2Series_id = T2Series_id
        self.mri_foci_distr_int = mri_foci_dist
        
    def __repr__(self):
        return "<Foci_record(lesion_id='%s', BenignNMaligNAnt='%s', DynSeries_id='%s', T2Series_id='%s', mri_foci_dist='%s')>" % (self.lesion_id, self.BenignNMaligNAnt, self.DynSeries_id, self.T2Series_id, self.mri_foci_distr_int)
           
           
#  created a Cad_record mapping 
class Dynamic_features(Base):
    """Base for Dynamic_features class using Declarative. for table f_dynamic"""
    __tablename__ = 'f_dynamic'
    __table_args__ = {'autoload':True}
    f_dyn_id = Column(Integer, primary_key=True)
    lesion_id = Column(Integer, ForeignKey('lesion.lesion_id'))
    
        
    def __init__(self, lesion_id, A_inside, alpha_inside, beta_inside, iAUC1_inside, Slope_ini_inside, Tpeak_inside,
                 Kpeak_inside, SER_inside, maxCr_inside, peakCr_inside, UptakeRate_inside, washoutRate_inside, maxVr_inside,
                 peakVr_inside, Vr_increasingRate_inside, Vr_decreasingRate_inside, Vr_post_1_inside,
                 A_countor, alpha_countor, beta_countor, iAUC1_countor, Slope_ini_countor, Tpeak_countor,
                 Kpeak_countor, SER_countor, maxCr_countor, peakCr_countor, UptakeRate_countor, washoutRate_countor, maxVr_countor,
                 peakVr_countor, Vr_increasingRate_countor, Vr_decreasingRate_countor, Vr_post_1_countor):
        self.lesion_id = lesion_id
        self.A_inside = A_inside
        self.alpha_inside = alpha_inside
        self.beta_inside = beta_inside
        self.iAUC1_inside = iAUC1_inside
        self.Slope_ini_inside = Slope_ini_inside
        self.Tpeak_inside = Tpeak_inside
        self.Kpeak_inside = Kpeak_inside
        self.SER_inside = SER_inside
        self.maxCr_inside = maxCr_inside
        self.peakCr_inside = peakCr_inside
        self.UptakeRate_inside = UptakeRate_inside
        self.washoutRate_inside = washoutRate_inside
        self.maxVr_inside = maxVr_inside
        self.peakVr_inside = peakVr_inside
        self.Vr_increasingRate_inside = Vr_increasingRate_inside
        self.Vr_decreasingRate_inside = Vr_decreasingRate_inside
        self.Vr_post_1_inside = Vr_post_1_inside
        self.A_countor = A_countor
        self.alpha_countor = alpha_countor
        self.beta_countor = beta_countor
        self.iAUC1_countor = iAUC1_countor
        self.Slope_ini_countor = Slope_ini_countor
        self.Tpeak_countor = Tpeak_countor
        self.Kpeak_countor = Kpeak_countor
        self.SER_countor = SER_countor
        self.maxCr_countor = maxCr_countor
        self.peakCr_countor = peakCr_countor
        self.UptakeRate_countor = UptakeRate_countor
        self.washoutRate_countor = washoutRate_countor
        self.maxVr_countor = maxVr_countor
        self.peakVr_countor = peakVr_countor
        self.Vr_increasingRate_countor = Vr_increasingRate_countor
        self.Vr_decreasingRate_countor = Vr_decreasingRate_countor
        self.Vr_post_1_countor = Vr_post_1_countor
        
        
#  created a Cad_record mapping 
class Morpho_features(Base):
    """Base for Morpho_features class using Declarative. for table f_morphology"""
    __tablename__ = 'f_morphology'
    __table_args__ = {'autoload':True}
    f_morpho_id = Column(Integer, primary_key=True)
    lesion_id = Column(Integer, ForeignKey('lesion.lesion_id'))
            
    def __init__(self, lesion_id, min_F_r_i, max_F_r_i, mean_F_r_i, var_F_r_i, skew_F_r_i, kurt_F_r_i,
                 iMax_Variance_uptake, iiMin_change_Variance_uptake, iiiMax_Margin_Gradient, k_Max_Margin_Grad,
                 ivVariance, circularity, irregularity, edge_sharp_mean, edge_sharp_std, max_RGH_mean, max_RGH_mean_k, max_RGH_var, max_RGH_var_k):
        self.lesion_id = lesion_id
        self.min_F_r_i = min_F_r_i
        self.max_F_r_i = max_F_r_i
        self.mean_F_r_i = mean_F_r_i
        self.var_F_r_i = var_F_r_i
        self.skew_F_r_i = skew_F_r_i
        self.kurt_F_r_i = kurt_F_r_i
        self.iMax_Variance_uptake = iMax_Variance_uptake
        self.iiMin_change_Variance_uptake = iiMin_change_Variance_uptake
        self.iiiMax_Margin_Gradient = iiiMax_Margin_Gradient
        self.k_Max_Margin_Grad = k_Max_Margin_Grad
        self.ivVariance = ivVariance
        self.circularity = circularity
        self.irregularity = irregularity
        self.edge_sharp_mean = edge_sharp_mean
        self.edge_sharp_std = edge_sharp_std
        self.max_RGH_mean = max_RGH_mean
        self.max_RGH_mean_k = max_RGH_mean_k
        self.max_RGH_var = max_RGH_var
        self.max_RGH_var_k = max_RGH_var_k        
        

#  created a Texture_features mapping 
class Texture_features(Base):
    """Base for Texture_features class using Declarative. for table f_texture"""
    __tablename__ = 'f_texture'
    __table_args__ = {'autoload':True}
    f_texture_id = Column(Integer, primary_key=True)
    lesion_id = Column(Integer, ForeignKey('lesion.lesion_id'))
        
    def __init__(self, lesion_id, texture_energy_nondir_post1, texture_contrast_nondir_post1, texture_correlation_nondir_post1,
                 texture_variance_nondir_post1, texture_inversediffmoment_nondir_post1, texture_sumaverage_nondir_post1,
                 texture_sumvariance_nondir_post1, texture_sumentropy_nondir_post1, texture_entropy_nondir_post1,
                 texture_diffvariance_nondir_post1,texture_diffentropy_nondir_post1,
                 texture_energy_nondir_post2, texture_contrast_nondir_post2, texture_correlation_nondir_post2, 
                 texture_variance_nondir_post2, texture_inversediffmoment_nondir_post2, texture_sumaverage_nondir_post2,
                 texture_sumvariance_nondir_post2, texture_sumentropy_nondir_post2, texture_entropy_nondir_post2, 
                 texture_diffvariance_nondir_post2, texture_diffentropy_nondir_post2, 
                 texture_energy_nondir_post3, texture_contrast_nondir_post3, texture_correlation_nondir_post3, 
                 texture_variance_nondir_post3, texture_inversediffmoment_nondir_post3, texture_sumaverage_nondir_post3,
                 texture_sumvariance_nondir_post3, texture_sumentropy_nondir_post3, texture_entropy_nondir_post3, 
                 texture_diffvariance_nondir_post3, texture_diffentropy_nondir_post3, 
                 texture_energy_nondir_post4, texture_contrast_nondir_post4, texture_correlation_nondir_post4, 
                 texture_variance_nondir_post4, texture_inversediffmoment_nondir_post4, texture_sumaverage_nondir_post4, 
                 texture_sumvariance_nondir_post4, texture_sumentropy_nondir_post4, texture_entropy_nondir_post4, 
                 texture_diffvariance_nondir_post4, texture_diffentropy_nondir_post4):
        self.lesion_id = lesion_id   
        self.texture_energy_nondir_post1=texture_energy_nondir_post1
        self.texture_contrast_nondir_post1=texture_contrast_nondir_post1
        self.texture_correlation_nondir_post1=texture_correlation_nondir_post1
        self.texture_variance_nondir_post1=texture_variance_nondir_post1
        self.texture_inversediffmoment_nondir_post1=texture_inversediffmoment_nondir_post1
        self.texture_sumaverage_nondir_post1=texture_sumaverage_nondir_post1
        self.texture_sumvariance_nondir_post1=texture_sumvariance_nondir_post1
        self.texture_sumentropy_nondir_post1=texture_sumentropy_nondir_post1 
        self.texture_entropy_nondir_post1=texture_entropy_nondir_post1
        self.texture_diffvariance_nondir_post1=texture_diffvariance_nondir_post1
        self.texture_diffentropy_nondir_post1=texture_diffentropy_nondir_post1
        self.texture_energy_nondir_post2=texture_energy_nondir_post2
        self.texture_contrast_nondir_post2=texture_contrast_nondir_post2
        self.texture_correlation_nondir_post2=texture_correlation_nondir_post2 
        self.texture_variance_nondir_post2=texture_variance_nondir_post2 
        self.texture_inversediffmoment_nondir_post2=texture_inversediffmoment_nondir_post2 
        self.texture_sumaverage_nondir_post2=texture_sumaverage_nondir_post2
        self.texture_sumvariance_nondir_post2=texture_sumvariance_nondir_post2
        self.texture_sumentropy_nondir_post2=texture_sumentropy_nondir_post2
        self.texture_entropy_nondir_post2=texture_entropy_nondir_post2
        self.texture_diffvariance_nondir_post2=texture_diffvariance_nondir_post2
        self.texture_diffentropy_nondir_post2=texture_diffentropy_nondir_post2
        self.texture_energy_nondir_post3=texture_energy_nondir_post3
        self.texture_contrast_nondir_post3=texture_contrast_nondir_post3 
        self.texture_correlation_nondir_post3=texture_correlation_nondir_post3
        self.texture_variance_nondir_post3=texture_variance_nondir_post3
        self.texture_inversediffmoment_nondir_post3=texture_inversediffmoment_nondir_post3
        self.texture_sumaverage_nondir_post3=texture_sumaverage_nondir_post3
        self.texture_sumvariance_nondir_post3=texture_sumvariance_nondir_post3
        self.texture_sumentropy_nondir_post3=texture_sumentropy_nondir_post3
        self.texture_entropy_nondir_post3=texture_entropy_nondir_post3
        self.texture_diffvariance_nondir_post3=texture_diffvariance_nondir_post3
        self.texture_diffentropy_nondir_post3=texture_diffentropy_nondir_post3
        self.texture_energy_nondir_post4=texture_energy_nondir_post4
        self.texture_contrast_nondir_post4=texture_contrast_nondir_post4
        self.texture_correlation_nondir_post4=texture_correlation_nondir_post4
        self.texture_variance_nondir_post4=texture_variance_nondir_post4
        self.texture_inversediffmoment_nondir_post4=texture_inversediffmoment_nondir_post4 
        self.texture_sumaverage_nondir_post4=texture_sumaverage_nondir_post4
        self.texture_sumvariance_nondir_post4=texture_sumvariance_nondir_post4 
        self.texture_sumentropy_nondir_post4=texture_sumentropy_nondir_post4
        self.texture_entropy_nondir_post4=texture_entropy_nondir_post4
        self.texture_diffvariance_nondir_post4=texture_diffvariance_nondir_post4
        self.texture_diffentropy_nondir_post4=texture_diffentropy_nondir_post4        


#  created a T2_features mapping 
class T2_features(Base):
    """Base for T2_features class using Declarative. for table f_T2"""
    __tablename__ = 'f_T2'
    __table_args__ = {'autoload':True}
    f_T2_id = Column(Integer, primary_key=True)
    lesion_id = Column(Integer, ForeignKey('lesion.lesion_id'))
            
    def __init__(self, lesion_id, find_t2_signal_int, T2dims, T2spacing, T2fatsat,  
                 T2_muscleSI, T2_muscleSIstd, muscle_scalar_range, bounds_muscleSI,
                 T2_lesionSI, T2_lesionSIstd, lesion_scalar_range, LMSIR,
                 T2min_F_r_i, T2max_F_r_i, T2mean_F_r_i, T2var_F_r_i, T2skew_F_r_i, T2kurt_F_r_i,
                 T2grad_margin, T2grad_margin_var, T2RGH_mean, T2RGH_var,
                 T2texture_energy_nondir, T2texture_contrast_nondir, T2texture_correlation_nondir,
                 T2texture_variance_nondir, T2texture_inversediffmoment_nondir, T2texture_sumaverage_nondir,
                 T2texture_sumvariance_nondir, T2texture_sumentropy_nondir, T2texture_entropy_nondir,
                 T2texture_diffvariance_nondir,T2texture_diffentropy_nondir):
        self.lesion_id = lesion_id   
        self.find_t2_signal_int= find_t2_signal_int
        self.T2dims= T2dims
        self.T2spacing= T2spacing
        self.T2fatsat= T2fatsat
        self.T2_muscleSI= T2_muscleSI 
        self.T2_muscleSIstd= T2_muscleSIstd
        self.muscle_scalar_range= muscle_scalar_range
        self.bounds_muscleSI= bounds_muscleSI
        self.T2_lesionSI= T2_lesionSI
        self.T2_lesionSIstd= T2_lesionSIstd
        self.lesion_scalar_range= lesion_scalar_range
        self.LMSIR= LMSIR
        self.T2min_F_r_i= T2min_F_r_i
        self.T2max_F_r_i= T2max_F_r_i
        self.T2mean_F_r_i= T2mean_F_r_i
        self.T2var_F_r_i= T2var_F_r_i
        self.T2skew_F_r_i= T2skew_F_r_i
        self.T2kurt_F_r_i= T2kurt_F_r_i
        self.T2grad_margin= T2grad_margin
        self.T2grad_margin_var= T2grad_margin_var
        self.T2RGH_mean= T2RGH_mean
        self.T2RGH_var= T2RGH_var
        self.T2texture_energy_nondir = T2texture_energy_nondir
        self.T2texture_contrast_nondir = T2texture_contrast_nondir
        self.T2texture_correlation_nondir = T2texture_correlation_nondir
        self.T2texture_variance_nondir= T2texture_variance_nondir
        self.T2texture_inversediffmoment_nondir= T2texture_inversediffmoment_nondir 
        self.T2texture_sumaverage_nondir= T2texture_sumaverage_nondir
        self.T2texture_sumvariance_nondir= T2texture_sumvariance_nondir
        self.T2texture_sumentropy_nondir= T2texture_sumentropy_nondir
        self.T2texture_entropy_nondir= T2texture_entropy_nondir
        self.T2texture_diffvariance_nondir= T2texture_diffvariance_nondir
        self.T2texture_diffentropy_nondir=T2texture_diffentropy_nondir



#  created a Annot_record mapping 
class Annot_record(Base):
    """Base for Annot_record class using Declarative. for table annotation
    attributes:
        self.lesion_id = lesion_id
        self.AccessionNumber = AccessionNumber
        self.SeriesDate = SeriesDate
        self.SeriesNumber = SeriesNumber
        self.SliceLocation = SliceLocation
        self.SeriesDescription = SeriesDescription
        self.PatientID = PatientID
        self.StudyID = StudyID
        self.note = note
        self.xi_coord = xi_coord
        self.yi_coord = yi_coord
        self.xf_coord = xf_coord
        self.yf_coord = yf_coord
        self.SeriesInstanceUID = SeriesInstanceUID
    """
    __tablename__ = 'annotation'
    __table_args__ = {'autoload':True}
    annot_id = Column(Integer, primary_key=True)
    lesion_id = Column(Integer, ForeignKey('lesion.lesion_id'))
        
    def __init__(self, lesion_id, AccessionNumber, SeriesDate, SeriesNumber, SliceLocation, SeriesDescription,
                 PatientID, StudyID, SeriesInstanceUID, note, xi_coord, yi_coord, xf_coord, yf_coord,
                 pi_ijk, pi_2display, pf_ijk, pf_2display, eu_dist_mkers, eu_dist_seg):
        self.lesion_id = lesion_id
        self.AccessionNumber = AccessionNumber
        self.SeriesDate = SeriesDate
        self.SeriesNumber = SeriesNumber
        self.SliceLocation = SliceLocation
        self.SeriesDescription = SeriesDescription
        self.PatientID = PatientID
        self.StudyID = StudyID
        self.SeriesInstanceUID = SeriesInstanceUID
        self.note = note
        self.xi_coord = xi_coord
        self.yi_coord = yi_coord
        self.xf_coord = xf_coord
        self.yf_coord = yf_coord
        self.pi_ijk = pi_ijk
        self.pi_2display = pi_2display
        self.pf_ijk = pf_ijk
        self.pf_2display = pf_2display
        self.eu_dist_mkers = eu_dist_mkers
        self.eu_dist_seg = eu_dist_seg
        
        
    def __repr__(self):
        return "<Annot_record(lesion_id='%s', PatientID='%s', StudyID='%s', note='%s', xi_coord='%s, yi_coord='%s', xf_coord='%s', yf_coord='%s')>" % (self.lesion_id, self.PatientID, self.StudyID, self.note, self.xi_coord, self.yi_coord, self.xf_coord, self.yf_coord)


#  created a Annot_record mapping 
class Segment_record(Base):
    """Base for Segment_record class using Declarative. for table segmentation
    attributes:
        self.lesion_id = lesion_id
        self.segm_xmin = segm_xmin
        self.segm_xmax = segm_xmax
        self.segm_ymin = segm_ymin
        self.segm_ymax = segm_ymax
        self.segm_zmin = segm_zmin
        self.segm_zmax = segm_zmax
        self.no_pts = no_pts
        self.voi_vol = voi_vol
        self.voi_surface = voi_surface
        self.lesion_centroid_world = lesion_centroid_world
        self.lesion_centroid_ijk = lesion_centroid_ijk
    """
    __tablename__ = 'segmentation'
    __table_args__ = {'autoload':True}
    segm_id = Column(Integer, primary_key=True)
    lesion_id = Column(Integer, ForeignKey('lesion.lesion_id'))
        
    def __init__(self, lesion_id, segm_xmin, segm_xmax, segm_ymin, segm_ymax, segm_zmin, segm_zmax,
                 no_pts, voi_vol, voi_surface, VOI_efect_diameter, lesion_centroid_world, lesion_centroid_ijk):
        self.lesion_id = lesion_id
        self.segm_xmin = segm_xmin
        self.segm_xmax = segm_xmax
        self.segm_ymin = segm_ymin
        self.segm_ymax = segm_ymax
        self.segm_zmin = segm_zmin
        self.segm_zmax = segm_zmax
        self.no_pts = no_pts
        self.voi_vol = voi_vol
        self.voi_surface = voi_surface
        self.voi_effect_dia = VOI_efect_diameter
        self.lesion_centroid_world = lesion_centroid_world
        self.lesion_centroid_ijk = lesion_centroid_ijk
        
    def __repr__(self):
        return "<Segment_record(lesion_id='%s', voi_vol='%s', voi_surface='%s', lesion_centroid_world='%s', lesion_centroid_ijk='%s')>" % (self.lesion_id, self.voi_vol, self.voi_surface, self.lesion_centroid_world, self.lesion_centroid_ijk)


#  created a Annot_record mapping 
class Stage1_record(Base):
    """Base for Stage1_record class using Declarative. for table segmentation
    attributes:
        self.lesion_id = lesion_id
        self.V0 = V0
        ...
        self.V19 = V19
    """
    __tablename__ = 'stage1features'
    __table_args__ = {'autoload':True}
    stage1_id = Column(Integer, primary_key=True)
    lesion_id = Column(Integer, ForeignKey('lesion.lesion_id'))
        
    def __init__(self, lesion_id, EdistV, earlySEV, dce2SEV, dce3SEV, lateSEV, ave_T2, network_meas):
        d_euclidean = [V for V in EdistV]
        earlySE = [V for V in earlySEV]
        dce2SE = [V for V in dce2SEV] 
        dce3SE = [V for V in dce3SEV] 
        lateSE = [V for V in lateSEV]
        self.degreeC = network_meas[0][0][1]
        self.closenessC = network_meas[1][0][1]
        self.betweennessC = network_meas[2][0][1]
        self.no_triangles = network_meas[3][0][1]
        self.no_con_comp = network_meas[4]
        self.lesion_id = lesion_id
        self.V0=d_euclidean[0]
        self.V1=d_euclidean[1]
        self.V2=d_euclidean[2]
        self.V3=d_euclidean[3]
        self.V4=d_euclidean[4]
        self.V5=d_euclidean[5]
        self.V6=d_euclidean[6]
        self.V7=d_euclidean[7]
        self.V8=d_euclidean[8]
        self.V9=d_euclidean[9]
        self.V10=d_euclidean[10]
        self.V11=d_euclidean[11]
        self.V12=d_euclidean[12]
        self.V13=d_euclidean[13]
        self.V14=d_euclidean[14]
        self.V15=d_euclidean[15]
        self.V16=d_euclidean[16]
        self.V17=d_euclidean[17]
        self.V18=d_euclidean[18]
        self.V19=d_euclidean[19]        
        self.earlySE0=earlySE[0]
        self.earlySE1=earlySE[1]
        self.earlySE2=earlySE[2]
        self.earlySE3=earlySE[3]
        self.earlySE4=earlySE[4]
        self.earlySE5=earlySE[5]
        self.earlySE6=earlySE[6]
        self.earlySE7=earlySE[7]
        self.earlySE8=earlySE[8]
        self.earlySE9=earlySE[9]
        self.earlySE10=earlySE[10]
        self.earlySE11=earlySE[11]
        self.earlySE12=earlySE[12]
        self.earlySE13=earlySE[13]
        self.earlySE14=earlySE[14]
        self.earlySE15=earlySE[15]
        self.earlySE16=earlySE[16]
        self.earlySE17=earlySE[17]
        self.earlySE18=earlySE[18]
        self.earlySE19=earlySE[19]  
        self.dce2SE0=dce2SE[0]
        self.dce2SE1=dce2SE[1]
        self.dce2SE2=dce2SE[2]
        self.dce2SE3=dce2SE[3]
        self.dce2SE4=dce2SE[4]
        self.dce2SE5=dce2SE[5]
        self.dce2SE6=dce2SE[6]
        self.dce2SE7=dce2SE[7]
        self.dce2SE8=dce2SE[8]
        self.dce2SE9=dce2SE[9]
        self.dce2SE10=dce2SE[10]
        self.dce2SE11=dce2SE[11]
        self.dce2SE12=dce2SE[12]
        self.dce2SE13=dce2SE[13]
        self.dce2SE14=dce2SE[14]
        self.dce2SE15=dce2SE[15]
        self.dce2SE16=dce2SE[16]
        self.dce2SE17=dce2SE[17]
        self.dce2SE18=dce2SE[18]
        self.dce2SE19=dce2SE[19] 
        self.dce3SE0=dce3SE[0]
        self.dce3SE1=dce3SE[1]
        self.dce3SE2=dce3SE[2]
        self.dce3SE3=dce3SE[3]
        self.dce3SE4=dce3SE[4]
        self.dce3SE5=dce3SE[5]
        self.dce3SE6=dce3SE[6]
        self.dce3SE7=dce3SE[7]
        self.dce3SE8=dce3SE[8]
        self.dce3SE9=dce3SE[9]
        self.dce3SE10=dce3SE[10]
        self.dce3SE11=dce3SE[11]
        self.dce3SE12=dce3SE[12]
        self.dce3SE13=dce3SE[13]
        self.dce3SE14=dce3SE[14]
        self.dce3SE15=dce3SE[15]
        self.dce3SE16=dce3SE[16]
        self.dce3SE17=dce3SE[17]
        self.dce3SE18=dce3SE[18]
        self.dce3SE19=dce3SE[19]
        self.lateSE0=lateSE[0]
        self.lateSE1=lateSE[1]
        self.lateSE2=lateSE[2]
        self.lateSE3=lateSE[3]
        self.lateSE4=lateSE[4]
        self.lateSE5=lateSE[5]
        self.lateSE6=lateSE[6]
        self.lateSE7=lateSE[7]
        self.lateSE8=lateSE[8]
        self.lateSE9=lateSE[9]
        self.lateSE10=lateSE[10]
        self.lateSE11=lateSE[11]
        self.lateSE12=lateSE[12]
        self.lateSE13=lateSE[13]
        self.lateSE14=lateSE[14]
        self.lateSE15=lateSE[15]
        self.lateSE16=lateSE[16]
        self.lateSE17=lateSE[17]
        self.lateSE18=lateSE[18]
        self.lateSE19=lateSE[19]
        # ave_T2
        self.ave_T20=ave_T2[0]
        self.ave_T21=ave_T2[1]
        self.ave_T22=ave_T2[2]
        self.ave_T23=ave_T2[3]
        self.ave_T24=ave_T2[4]
        self.ave_T25=ave_T2[5]
        self.ave_T26=ave_T2[6]
        self.ave_T27=ave_T2[7]
        self.ave_T28=ave_T2[8]
        self.ave_T29=ave_T2[9]
        self.ave_T210=ave_T2[10]
        self.ave_T211=ave_T2[11]
        self.ave_T212=ave_T2[12]
        self.ave_T213=ave_T2[13]
        self.ave_T214=ave_T2[14]
        self.ave_T215=ave_T2[15]
        self.ave_T216=ave_T2[16]
        self.ave_T217=ave_T2[17]
        self.ave_T218=ave_T2[18]
        self.ave_T219=ave_T2[19]
        
        
    def __repr__(self):
        return "<Stage1_record(lesion_id='%s', V0='%s', V1='%s'')>" % (self.lesion_id, self.V0, self.V1)
        
        
#  created a Annot_record mapping 
class Radiology_record(Base):
    """Base for Radiology_record class using Declarative. for table Radiology_record
    attributes:
        self.lesion_id = lesion_id
        self.cad_pt_no_txt = 
        self.latest_mutation = 
        self.exam_dt_datetime
        self.mri_cad_status_txt 
        self.comment_txt
        self.radiology_rpt_generated_yn
        self.original_report_txt
        self.sty_indicator_rout_screening_obsp_yn
        self.sty_indicator_high_risk_yn
        self.sty_indicator_high_risk_brca_1_yn
        self.sty_indicator_high_risk_brca_2_yn
        self.sty_indicator_high_risk_brca_1_or_2_yn
        self.sty_indicator_high_risk_at_yn
        self.sty_indicator_high_risk_other_gene_yn
        self.sty_indicator_high_risk_prior_high_risk_marker_yn
        self.sty_indicator_high_risk_prior_personal_can_hist_yn
        self.sty_indicator_high_risk_hist_of_mantle_rad_yn
        self.sty_indicator_high_risk_fam_hist_yn
        self.sty_indicator_add_eval_as_folup_yn
        self.sty_indicator_folup_after_pre_exam_yn
        self.sty_indicator_pre_operative_extent_of_dis_yn
        self.sty_indicator_post_operative_margin_yn
        self.sty_indicator_pre_neoadj_trtmnt_yn
        self.sty_indicator_prob_solv_diff_img_yn
        self.sty_indicator_scar_vs_recurr_yn
        self.sty_indicator_folup_recommend_yn
        self.sty_indicator_prior_2_prophy_mast_yn
    """
    __tablename__ = 'radiologyInfo'
    __table_args__ = {'autoload':True}
    radio_id = Column(Integer, primary_key=True)
    lesion_id = Column(Integer, ForeignKey('lesion.lesion_id'))
        
    def __init__(self, lesion_id, cad_pt_no_txt, latest_mutation, exam_dt_datetime, mri_cad_status_txt, comment_txt,
                 original_report_txt,
                 sty_indicator_rout_screening_obsp_yn, sty_indicator_high_risk_yn,
                 sty_indicator_high_risk_brca_1_yn, sty_indicator_high_risk_brca_2_yn, sty_indicator_high_risk_brca_1_or_2_yn,
                 sty_indicator_high_risk_at_yn, sty_indicator_high_risk_other_gene_yn, sty_indicator_high_risk_prior_high_risk_marker_yn,
                 sty_indicator_high_risk_prior_personal_can_hist_yn, sty_indicator_high_risk_hist_of_mantle_rad_yn,
                 sty_indicator_high_risk_fam_hist_yn, sty_indicator_add_eval_as_folup_yn, sty_indicator_folup_after_pre_exam_yn,
                 sty_indicator_pre_operative_extent_of_dis_yn, sty_indicator_post_operative_margin_yn, sty_indicator_pre_neoadj_trtmnt_yn,
                 sty_indicator_prob_solv_diff_img_yn, sty_indicator_scar_vs_recurr_yn, sty_indicator_folup_recommend_yn,
                 sty_indicator_prior_2_prophy_mast_yn):
        
        self.lesion_id = lesion_id
        self.cad_pt_no_txt = cad_pt_no_txt
        self.latest_mutation = latest_mutation
        self.exam_dt_datetime = exam_dt_datetime
        self.mri_cad_status_txt = mri_cad_status_txt
        self.comment_txt = comment_txt
        self.original_report_txt = original_report_txt
        self.sty_indicator_rout_screening_obsp_yn = sty_indicator_rout_screening_obsp_yn
        self.sty_indicator_high_risk_yn = sty_indicator_high_risk_yn
        self.sty_indicator_high_risk_brca_1_yn = sty_indicator_high_risk_brca_1_yn
        self.sty_indicator_high_risk_brca_2_yn = sty_indicator_high_risk_brca_2_yn
        self.sty_indicator_high_risk_brca_1_or_2_yn = sty_indicator_high_risk_brca_1_or_2_yn
        self.sty_indicator_high_risk_at_yn = sty_indicator_high_risk_at_yn
        self.sty_indicator_high_risk_other_gene_yn = sty_indicator_high_risk_other_gene_yn
        self.sty_indicator_high_risk_prior_high_risk_marker_yn = sty_indicator_high_risk_prior_high_risk_marker_yn
        self.sty_indicator_high_risk_prior_personal_can_hist_yn = sty_indicator_high_risk_prior_personal_can_hist_yn
        self.sty_indicator_high_risk_hist_of_mantle_rad_yn = sty_indicator_high_risk_hist_of_mantle_rad_yn
        self.sty_indicator_high_risk_fam_hist_yn = sty_indicator_high_risk_fam_hist_yn
        self.sty_indicator_add_eval_as_folup_yn = sty_indicator_add_eval_as_folup_yn
        self.sty_indicator_folup_after_pre_exam_yn = sty_indicator_folup_after_pre_exam_yn
        self.sty_indicator_pre_operative_extent_of_dis_yn = sty_indicator_pre_operative_extent_of_dis_yn
        self.sty_indicator_post_operative_margin_yn = sty_indicator_post_operative_margin_yn
        self.sty_indicator_pre_neoadj_trtmnt_yn = sty_indicator_pre_neoadj_trtmnt_yn
        self.sty_indicator_prob_solv_diff_img_yn = sty_indicator_prob_solv_diff_img_yn
        self.sty_indicator_scar_vs_recurr_yn = sty_indicator_scar_vs_recurr_yn
        self.sty_indicator_folup_recommend_yn = sty_indicator_folup_recommend_yn
        self.sty_indicator_prior_2_prophy_mast_yn = sty_indicator_prior_2_prophy_mast_yn
        
        
    def __repr__(self):
        return "<Radiology_record(lesion_id='%s', original_report_txt='%s')>" % (self.lesion_id, self.original_report_txt)        

