# -*- coding: utf-8 -*-
"""
Created on Fri Apr 01 16:11:43 2016

@author: rmaani
"""

# -*- coding: utf-8 -*-
"""
Purpose:
based on Cristina's verified groud truth list:
    Z:\\Breast\\gtSeg_verified\\CADlesions_wSeries.csv'
run cad pipeline.

@author: yingli
"""

import pandas as pd
import os
import os.path
import shutil
import glob
import tempfile
import SimpleITK as sitk

def do_elastix_registration(wofs_filename_full,pre_contrast_filename,elastix_pars_affine,elastix_temp_output_path,wofs_reg_filename_full):
    elastix_command = 'elastix -f {}  -m {} -p {}  -out {} -threads 4'.format(pre_contrast_filename,wofs_filename_full,elastix_pars_affine,elastix_temp_output_path);
    print elastix_command
    os.system(elastix_command)
    source = os.path.join(elastix_temp_output_path,  "result.0.mha")
    dst = wofs_reg_filename_full
    shutil.move(source,dst)
    
def do_breastsegmentation(wofs_filename_full, breast_mask_filename_full):
    breastsegmentation_cmd = 'breastsegmentation -i {}  -x 1 -y 1 -z 1 -o {}'.format(wofs_filename_full, breast_mask_filename_full)
    print breastsegmentation_cmd
    os.system(breastsegmentation_cmd)
 
def do_motion_correstion(dynamic_filename_full):
    motioncorrection_command = "motioncorrection -f {}  -m {} {} {} {}".format( dynamic_filename_full[0],dynamic_filename_full[1],dynamic_filename_full[2],dynamic_filename_full[3],dynamic_filename_full[4] ) 
    print motioncorrection_command
    os.system(motioncorrection_command)    
    
def do_lesion_segmentation(dynamic_filename_mc_full,breast_mask_filename_full,lesion_segmentation_output_prefix):
    lesionsegmentation_command = "lesionsegmentation -i {} {} {} {} {} -m {} -o {}".format(dynamic_filename_mc_full[0],dynamic_filename_mc_full[1],dynamic_filename_mc_full[2],dynamic_filename_mc_full[3],dynamic_filename_mc_full[4],breast_mask_filename_full,lesion_segmentation_output_prefix)
    print lesionsegmentation_command
    os.system(lesionsegmentation_command)
    
input_path = 'd:\\ground_truth_mha'
output_path = 'd:\\ground_truth_mha_pipeline_intermediate'
if not os.path.exists(output_path):
    os.mkdir(output_path)

dicom_path = 'Z:\\Breast\\DICOMS\\'
lesion_ground_truth_path = 'Z:\\Breast\\gtSeg_verified'
lesion_ground_truth_master_lister ='CADlesions_wSeries.csv'
cad_lesions_csv = pd.read_csv(os.path.join(lesion_ground_truth_path,lesion_ground_truth_master_lister));

lesion_ids = cad_lesions_csv['lesion_id']
patient_ids = cad_lesions_csv['cad_pt_no_txt']
access_numbers = cad_lesions_csv['exam_a_number_txt']
pre_contrast_series_nums = cad_lesions_csv['DynSeries_id']

f = open( output_path + os.sep + 'cad_pipeline_error.txt','w')

#elastix parameters
elastix_pars_affine = os.path.join(r'C:\my_bin_tools\breast_MR_utils\bin', 'elastix_pars_affine.txt')
elastix_temp_output_path = tempfile.gettempdir() 
print elastix_temp_output_path

for i in range(len(lesion_ids)):
    try:
        lesionid_patientid_accessnumber = '{}_{}_{}'.format(str(lesion_ids[i]),str(patient_ids[i]).zfill(4),str(access_numbers[i]))
        #get wofs filename
        wofs_filename = '{}_{}_{}_wofs.mha'.format(str(lesion_ids[i]),str(patient_ids[i]).zfill(4),str(access_numbers[i]))
        wofs_filename_full = os.path.join(output_path,wofs_filename)
         
        #get dynamics filename
        dynamic_filename_full_source=[]
        precontrast_series_num = int(str(pre_contrast_series_nums[i][1:])) #s600, [1:] remove the 's'
        dynamic_series_nums = [str(n) for n in range(precontrast_series_num,precontrast_series_num+5)]
        for j in range(5):
            #the output mha:lesionid_patientid_access#_series#@acqusionTime.mha
            dynamic_filename = '{}_{}_{}_{}'.format( str(lesion_ids[i]), str(patient_ids[i]).zfill(4), str(access_numbers[i]), dynamic_series_nums[j] )
            
            #write log if mha file not exist             
            glob_result = glob.glob(os.path.join(input_path,dynamic_filename+'*')) #'*':do not to know the exactly acquistion time
            if glob_result != []:
               dynamic_filename_full_source.append(glob_result[0])
        
        wofs_img = sitk.ReadImage(wofs_filename_full)
        wofs_img_size = wofs_img.GetSize()
#        print wofs_filename_full,wofs_img_size
#        continue
    
        if wofs_img_size[2] > 110:
             f.write(lesionid_patientid_accessnumber)
             f.write('\n')
             f.flush()
             continue
            
        #copy to output path
      
        wofs_filename_full_source = os.path.join(input_path,wofs_filename)
        shutil.copy(wofs_filename_full_source,wofs_filename_full)
    
        dynamic_filename_full = dynamic_filename_full_source[:]
        for i in range(5):
            dynamic_filename_full[i]=os.path.join(output_path,os.path.basename(dynamic_filename_full[i]))
            shutil.copy(dynamic_filename_full_source[i],dynamic_filename_full[i])
            
           
        #elastix registration:
        if os.path.exists(wofs_filename_full) and os.path.exists(dynamic_filename_full[0]):
            wofs_reg_filename_full = os.path.join(output_path,os.path.splitext(wofs_filename)[0] + '_reg.mha')
            do_elastix_registration(wofs_filename_full,dynamic_filename_full[0],elastix_pars_affine,elastix_temp_output_path,wofs_reg_filename_full)

        #breast segmentation    
        if os.path.exists(wofs_reg_filename_full):
            breast_mask_filename_full = os.path.splitext(wofs_reg_filename_full)[0]+"_breast_mask.mha"
            do_breastsegmentation(wofs_reg_filename_full, breast_mask_filename_full)
        
#        #motion correction
        if os.path.exists(dynamic_filename_full[0]) and \
           os.path.exists(dynamic_filename_full[1]) and \
           os.path.exists(dynamic_filename_full[2]) and \
           os.path.exists(dynamic_filename_full[3]) and \
           os.path.exists(dynamic_filename_full[4]):
           do_motion_correstion(dynamic_filename_full)
       
        #lesion segmenation
        dynamic_filename_mc_full = dynamic_filename_full[:]
        for i in range(1,5):
            dynamic_filename_mc_full[i]=os.path.splitext(dynamic_filename_full[i])[0]+"_mc.mha"
        lesion_segmentation_output_prefix = os.path.join(output_path, lesionid_patientid_accessnumber+'_lesion_segmentation')
        
        if os.path.exists(dynamic_filename_mc_full[0]) and \
           os.path.exists(dynamic_filename_mc_full[1]) and \
           os.path.exists(dynamic_filename_mc_full[2]) and \
           os.path.exists(dynamic_filename_mc_full[3]) and \
           os.path.exists(dynamic_filename_mc_full[4]) and \
           os.path.exists(breast_mask_filename_full):
           do_lesion_segmentation(dynamic_filename_mc_full,breast_mask_filename_full,lesion_segmentation_output_prefix)
           
    except:
        print "exception"
        f.write(lesionid_patientid_accessnumber)
        f.write('\n')
        f.flush()
        
        