import pandas as pd
import numpy as np
import matplotlib as plt
from linearmodels import PanelOLS
import statsmodels.api as sm
import econtools as econ
import econtools.metrics as mt
from statsmodels.stats.outliers_influence import variance_inflation_factor

from auxiliary.prepare import *

def table2_col1(data):
    df =data
    df_reg_co = df[(df['turin_co_sample']==1)&(df['ctrl_exp_turin_co_sample']==1)&(df['post_experience']>= 5) & (df['pre_experience']>=5) &(df['post_experience'].isnull() == False ) & 
                   (df['pre_experience'].isnull()==False)&(df['missing']==0)]
    outcome = ['discount', 'delay_ratio', 'overrun_ratio', 'days_to_award']

    #iteration
    for o in outcome:
        idx = df_reg_co[df_reg_co[o].isnull()==True].index
        df_name = df_reg_co.drop(idx)
    
        #make a column list
        reg_col = []
        for j in year_list:
            reg_col.append(j)
        exog_var = ['fpsb_auction']
        exog = exog_var + reg_col 


        #check multicollinearity
        X = df_name.loc[:,exog]
        vif = calc_vif(X)

        #delete from col list
        for i in range(len(vif)):
            if np.isnan(vif.loc[i, 'VIF']) == True:
                reg_col.remove(vif.loc[i, 'variables'])
                
        exog = exog_var + reg_col
        exog.remove(2000)

        fe_reg = mt.reg(df_name, o, exog, fe_name = 'authority_code', cluster = 'auth_anno')
        return(fe_reg)
    
def table2_col2(data):
    df = data
    df_reg_co = df[(df['turin_co_sample']==1)&(df['ctrl_exp_turin_co_sample']==1)&(df['post_experience']>= 5) & (df['pre_experience']>=5) &(df['post_experience'].isnull() == False ) &
                   (df['pre_experience'].isnull()==False)&(df['missing']==0)]
    outcome = ['discount', 'delay_ratio', 'overrun_ratio', 'days_to_award']

    #iteration
    for o in outcome:
        idx = df_reg_co[df_reg_co[o].isnull()==True].index
        df_name = df_reg_co.drop(idx)
    
        #make a column list
        reg_col = []
        for i in work_list:
            reg_col.append(i)
        for j in year_list:
            reg_col.append(j)
        exog_var = ['fpsb_auction','id_auth','reserve_price','municipality']
        exog = exog_var + reg_col 

        #check multicollinearity
        X = df_name.loc[:,exog]
        vif = calc_vif(X)

        #delete from col list
        for i in range(len(vif)):
            if np.isnan(vif.loc[i, 'VIF']) == True:
                reg_col.remove(vif.loc[i, 'variables'])
            elif vif.loc[i,'VIF'] > 10:
                for j in exog_var:
                    if str(vif.loc[i,'variables']) is j and vif.loc[i,'variables'] is not 'fpsb_auction' and vif.loc[i,'variables'] is not 'id_auth':
                        exog_var.remove(vif.loc[i,'variables'])
                
        exog = exog_var + reg_col
        exog.remove('id_auth')
        exog.remove(2000)
        exog.remove('OG01')

        #print(exog) #check
        fe_reg = mt.reg(df_name, o, exog, fe_name = 'authority_code', cluster = 'auth_anno')
        return(fe_reg)

def table2_col3(data):
    df = data
    df_reg_co = df[(df['turin_co_sample']==1)&(df['ctrl_pop_turin_co_sample']==1)&(df['post_experience']>= 5) & (df['pre_experience']>=5) &(df['post_experience'].isnull() == False ) & 
                   (df['pre_experience'].isnull()==False)&(df['missing']==0)]
    outcome = ['discount', 'delay_ratio', 'overrun_ratio', 'days_to_award']

    #iteration
    for o in outcome:
        idx = df_reg_co[df_reg_co[o].isnull()==True].index
        df_name = df_reg_co.drop(idx)
    
        #make a column list
        reg_col = []
        for j in year_list:
            reg_col.append(j)
        exog_var = ['fpsb_auction']
        exog = exog_var + reg_col 


        #check multicollinearity
        X = df_name.loc[:,exog]
        vif = calc_vif(X)

        #delete from col list
        for i in range(len(vif)):
            if np.isnan(vif.loc[i, 'VIF']) == True:
                reg_col.remove(vif.loc[i, 'variables'])
                
        exog = exog_var + reg_col
        exog.remove(2000)

        fe_reg = mt.reg(df_name, o, exog, fe_name = 'authority_code', cluster = 'auth_anno')
        return(fe_reg)
    
def table2_col4(data):
    df = data
    df_reg_co = df[(df['turin_co_sample']==1)&(df['ctrl_pop_turin_co_sample']==1)&(df['post_experience']>= 5) & (df['pre_experience']>=5) &(df['post_experience'].isnull() == False ) & 
                   (df['pre_experience'].isnull()==False)&(df['missing']==0)]
    outcome = ['discount', 'delay_ratio', 'overrun_ratio', 'days_to_award']

    for o in outcome:
        idx = df_reg_co[df_reg_co[o].isnull()==True].index
        df_name = df_reg_co.drop(idx)
        
        #to make a column list
        reg_col = []
        for i in work_list:
            reg_col.append(i)
        for j in year_list:
            reg_col.append(j)
        exog_var = ['fpsb_auction','reserve_price','municipality']
        exog = exog_var + reg_col 

        #check multicollinearity
        X = df_name.loc[:,exog]
        vif = calc_vif(X)

        #delete from col list
        for i in range(len(vif)):
            if np.isnan(vif.loc[i, 'VIF']) == True:
                reg_col.remove(vif.loc[i, 'variables'])
            elif vif.loc[i,'VIF'] > 10:
                for j in exog_var:
                    if str(vif.loc[i,'variables']) is j and vif.loc[i,'variables'] is not 'fpsb_auction' and vif.loc[i,'variables'] is not 'id_auth':
                        exog_var.remove(vif.loc[i,'variables'])
                
        exog = exog_var + reg_col
        exog.remove(2000)
        exog.remove('OG01')
        exog.remove('municipality')

        fe_reg = mt.reg(df_name, o, exog, fe_name = 'authority_code', cluster = 'auth_anno')
        return(fe_reg)
    
def table2_col5(data):
    df = data
    df_reg_co = df[(df['turin_co_sample']==1)&(df['ctrl_pop_exp_turin_co_sample']==1)&(df['post_experience']>= 5) & (df['pre_experience']>=5) &(df['post_experience'].isnull() == False ) & 
                   (df['pre_experience'].isnull()==False)&(df['missing']==0)]
    outcome = ['discount', 'delay_ratio', 'overrun_ratio', 'days_to_award']

    #iteration
    for o in outcome:
        idx = df_reg_co[df_reg_co[o].isnull()==True].index
        df_name = df_reg_co.drop(idx)
    
        #make a column list
        reg_col = []
        for j in year_list:
            reg_col.append(j)
        exog_var = ['fpsb_auction']
        exog = exog_var + reg_col 


        #check multicollinearity
        X = df_name.loc[:,exog]
        vif = calc_vif(X)

        #delete from col list
        for i in range(len(vif)):
            if np.isnan(vif.loc[i, 'VIF']) == True:
                reg_col.remove(vif.loc[i, 'variables'])
                
        exog = exog_var + reg_col
        exog.remove(2000)

        fe_reg = mt.reg(df_name, o, exog, fe_name = 'authority_code', cluster = 'auth_anno')
        return(fe_reg)    
    
def table2_col6(data):
    df = data
    df_reg_co = df[(df['turin_co_sample']==1)&(df['ctrl_pop_exp_turin_co_sample']==1)&(df['post_experience']>= 5) & (df['pre_experience']>=5) &(df['post_experience'].isnull() == False ) & 
                   (df['pre_experience'].isnull()==False)&(df['missing']==0)]
    outcome = ['discount', 'delay_ratio', 'overrun_ratio', 'days_to_award']

    #iteration
    for o in outcome:
        idx = df_reg_co[df_reg_co[o].isnull()==True].index
        df_name = df_reg_co.drop(idx)

        #make a column list
        reg_col = []
        for i in work_list:
            reg_col.append(i)
        for j in year_list:
            reg_col.append(j)
        exog_var = ['fpsb_auction','reserve_price','municipality']
        exog = exog_var + reg_col 

        #check multicollinearity
        X = df_name.loc[:,exog]
        vif = calc_vif(X)

        #delete from col list
        for i in range(len(vif)):
            if np.isnan(vif.loc[i, 'VIF']) == True:
                reg_col.remove(vif.loc[i, 'variables'])
            elif vif.loc[i,'VIF'] > 10:
                for j in exog_var:
                    if str(vif.loc[i,'variables']) is j and vif.loc[i,'variables'] is not 'fpsb_auction' and vif.loc[i,'variables'] is not 'id_auth':
                        exog_var.remove(vif.loc[i,'variables'])
                
        exog = exog_var + reg_col
        exog.remove(2000)
        exog.remove('OG01')
        exog.remove('municipality')

        fe_reg = mt.reg(df_name, o, exog, fe_name = 'authority_code', cluster = 'auth_anno')
        return(fe_reg)
        
        