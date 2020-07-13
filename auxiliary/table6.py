import pandas as pd
import numpy as np
import matplotlib as plt
from linearmodels import PanelOLS
import statsmodels.api as sm
import econtools as econ
import econtools.metrics as mt
from statsmodels.stats.outliers_influence import variance_inflation_factor

from auxiliary.prepare import *
from auxiliary.table2 import *
from auxiliary.table3 import *
from auxiliary.table4 import *
from auxiliary.table6 import *
from auxiliary.table_formula import *

def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

def table6_setting(data):
    df = data
    work_list = df['work_category'].unique()
    year_list = df['year'].unique()
    df['post01pre05'] = 0
    for i in range(len(df)):
        if df.loc[i,'year'] >= 2001 and df.loc[i,'year'] <= 2005 and pd.isnull(df.loc[i,'year']) == False:
            df.loc[i,'post01pre05'] = 1

    df['post02pre04'] = 0
    for i in range(len(df)):
        if df.loc[i,'year'] >= 2002 and df.loc[i,'year'] <= 2004 and pd.isnull(df.loc[i,'year']) == False:
            df.loc[i,'post02pre04'] = 1

    df['complexity_dummy'] = 0
    for i in range(len(df)):
        if df.loc[i,'complex_work'] ==2 or df.loc[i,'complex_work']==3:
            df.loc[i, 'complexity_dummy'] = 1
    
    complex_list = df['year'].unique()
    
    return(df)

def table6_col1(data):
    df = data
    work_list = df['work_category'].unique()
    year_list = df['year'].unique()
    treatment = ['turin_co_sample','turin_pr_sample']
    
    for t in treatment:
        df_reg_co = df[(df[t]==1)&(df['ctrl_exp_' + t]==1)&(df['post_experience']>= 5) & (df['pre_experience']>=5) &(df['post_experience'].isnull() == False ) & (df['pre_experience'].isnull()==False)&
                       (df['missing']==0)]

        #vif cal
        #first, make a column list
        reg_col = []
        for i in work_list:
            reg_col.append(i)
        for j in year_list:
            reg_col.append(j)
        exog_var = ['fpsb_auction','reserve_price','municipality'] #id_auth빼봄
        exog = exog_var + reg_col 


        #check multicollinearity
        X = df_reg_co.loc[:,exog]
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

        if t == 'turin_co_sample':
            fe_reg_co = mt.reg(df_reg_co, 'discount', exog, fe_name = 'authority_code', cluster = 'auth_anno')
        else:
            fe_reg_pr = mt.reg(df_reg_co, 'discount', exog, fe_name = 'authority_code', cluster = 'auth_anno')
    
    fe_reg = (fe_reg_co, fe_reg_pr )
    return(fe_reg)


def tabble6_col2(data):
    df = data
    work_list = df['work_category'].unique()
    year_list = df['year'].unique()
    treatment = ['turin_co_sample','turin_pr_sample']

    for t in treatment:
        df_reg_co = df[(df[t+'_full']==1)&(df['ctrl_exp_' + t]==1)&(df['post_experience']>= 5) & (df['pre_experience']>=5) &(df['post_experience'].isnull() == False ) & 
                       (df['pre_experience'].isnull()==False)&(df['missing']==0)]

        #first, make a column list
        reg_col = []
        for i in work_list:
            reg_col.append(i)
        for j in year_list:
            reg_col.append(j)
        exog_var = ['fpsb_auction','reserve_price','municipality'] #id_auth빼봄
        exog = exog_var + reg_col 


        #check multicollinearity
        X = df_reg_co.loc[:,exog]
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

        if t == 'turin_co_sample':
            fe_reg_co = mt.reg(df_reg_co, 'discount', exog, fe_name = 'authority_code', cluster = 'auth_anno')
        else:
            fe_reg_pr = mt.reg(df_reg_co, 'discount', exog, fe_name = 'authority_code', cluster = 'auth_anno')
    
    fe_reg = (fe_reg_co, fe_reg_pr )
    return(fe_reg)


def table6_col3(data):
    data = df
    work_list = df['work_category'].unique()
    year_list = df['year'].unique()
    treatment = ['turin_co_sample','turin_pr_sample']

    for t in treatment:
        df_reg_co = df[(df[t]==1)&(df['ctrl_exp_'+t] == 1)&(df['post_experience']>= 5) & (df['pre_experience']>=5) &(df['post_experience'].isnull() == False ) & (df['pre_experience'].isnull()==False)&
                       (df['missing']==1)]

        #first, make a column list
        reg_col = []
        for i in work_list:
            reg_col.append(i)
        for j in year_list:
            reg_col.append(j)
        exog_var = ['fpsb_auction','reserve_price','municipality'] #id_auth빼봄
        exog = exog_var + reg_col 


        #check multicollinearity
        X = df_reg_co.loc[:,exog]
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

        if t == 'turin_co_sample':
            fe_reg_co = mt.reg(df_reg_co, 'discount', exog, fe_name = 'authority_code', cluster = 'auth_anno')
        else:
            fe_reg_pr = mt.reg(df_reg_co, 'discount', exog, fe_name = 'authority_code', cluster = 'auth_anno')
    
    fe_reg = (fe_reg_co, fe_reg_pr )
    return(fe_reg)


def table6_col4(data):
    df = data
    work_list = df['work_category'].unique()
    year_list = df['year'].unique()
    treatment = ['turin_co_sample','turin_pr_sample']

    for t in treatment:
        df_reg_co = df[(df['complexity_dummy']==0)&(df[t]==1)&(df['ctrl_exp_'+t] == 1)&(df['post_experience']>= 5) & (df['pre_experience']>=5) &(df['post_experience'].isnull() == False ) & 
                       (df['pre_experience'].isnull()==False)&
                       (df['missing']==0)]

        #first, make a column list
        reg_col = []
        for i in work_list:
            reg_col.append(i)
        for j in year_list:
            reg_col.append(j)
        exog_var = ['fpsb_auction','reserve_price','municipality'] #id_auth빼봄
        exog = exog_var + reg_col 


        #check multicollinearity
        X = df_reg_co.loc[:,exog]
        vif = calc_vif(X)
        #print(vif)


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
        #exog.remove('OG01')
        exog.remove('municipality')

        exog.remove('OS24')

        if t == 'turin_co_sample':
            fe_reg_co = mt.reg(df_reg_co, 'discount', exog, fe_name = 'authority_code', cluster = 'auth_anno')
        else:
            fe_reg_pr = mt.reg(df_reg_co, 'discount', exog, fe_name = 'authority_code', cluster = 'auth_anno')
    
    fe_reg = (fe_reg_co, fe_reg_pr )
    return(fe_reg)


def table6_col5(data):
    df = data
    work_list = df['work_category'].unique()
    year_list = df['year'].unique()
    treatment = ['turin_co_sample','turin_pr_sample']

    for t in treatment:
        df_reg_co = df[(df['complexity_dummy']==1)&(df[t]==1)&(df['ctrl_exp_'+t] == 1)&(df['post_experience']>= 5) & (df['pre_experience']>=5) &(df['post_experience'].isnull() == False ) & 
                       (df['pre_experience'].isnull()==False)&
                       (df['missing']==0)]

        #first, make a column list
        reg_col = []
        for i in work_list:
            reg_col.append(i)
        for j in year_list:
            reg_col.append(j)
        exog_var = ['fpsb_auction','reserve_price','municipality'] #id_auth빼봄
        exog = exog_var + reg_col 


        #check multicollinearity
        X = df_reg_co.loc[:,exog]
        vif = calc_vif(X)
        #print(vif)


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

        if t == 'turin_co_sample':
            fe_reg_co = mt.reg(df_reg_co, 'discount', exog, fe_name = 'authority_code', cluster = 'auth_anno')
        else:
            fe_reg_pr = mt.reg(df_reg_co, 'discount', exog, fe_name = 'authority_code', cluster = 'auth_anno')
    
    fe_reg = (fe_reg_co, fe_reg_pr )
    return(fe_reg)


def table6_col6(data):
    data = df
    work_list = df['work_category'].unique()
    year_list = df['year'].unique()
    treatment = ['turin_co_sample','turin_pr_sample']

    for t in treatment:
        df_reg_co = df[(df['post01pre05']==1)&(df[t]==1)&(df['ctrl_exp_'+t] == 1)&(df['post_experience']>= 5) & (df['pre_experience']>=5) &(df['post_experience'].isnull() == False ) & 
                       (df['pre_experience'].isnull()==False)&
                       (df['missing']==0)]

        #first, make a column list
        reg_col = []
        for i in work_list:
            reg_col.append(i)
        for j in year_list:
            reg_col.append(j)
        exog_var = ['fpsb_auction','reserve_price','municipality'] #id_auth빼봄
        exog = exog_var + reg_col 


        #check multicollinearity
        X = df_reg_co.loc[:,exog]
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
        exog.remove('OG01')
        exog.remove('municipality')

        exog.remove(2005)

        if t == 'turin_co_sample':
            fe_reg_co = mt.reg(df_reg_co, 'discount', exog, fe_name = 'authority_code', cluster = 'auth_anno')
        else:
            fe_reg_pr = mt.reg(df_reg_co, 'discount', exog, fe_name = 'authority_code', cluster = 'auth_anno')
    
    fe_reg = (fe_reg_co, fe_reg_pr )
    return(fe_reg)


def table6_col7(data):
    df= data
    work_list = df['work_category'].unique()
    year_list = df['year'].unique()
    treatment = ['turin_co_sample','turin_pr_sample']

    for t in treatment:
        df_reg_co = df[(df['post02pre04']==1)&(df[t]==1)&(df['ctrl_exp_'+t] == 1)&(df['post_experience']>= 5) & (df['pre_experience']>=5) &(df['post_experience'].isnull() == False ) & 
                       (df['pre_experience'].isnull()==False)&
                       (df['missing']==0)]

        #vif cal
        #first, make a column list
        reg_col = []
        for i in work_list:
            reg_col.append(i)
        for j in year_list:
            reg_col.append(j)
        exog_var = ['fpsb_auction','reserve_price','municipality'] #id_auth빼봄
        exog = exog_var + reg_col 


        #check multicollinearity
        X = df_reg_co.loc[:,exog]
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
        #exog.remove(2000)
        exog.remove('OG01')
        exog.remove('municipality')

        exog.remove(2004)
        
        if t == 'turin_co_sample':
            fe_reg_co = mt.reg(df_reg_co, 'discount', exog, fe_name = 'authority_code', cluster = 'auth_anno')
        else:
            fe_reg_pr = mt.reg(df_reg_co, 'discount', exog, fe_name = 'authority_code', cluster = 'auth_anno')
    
    fe_reg = (fe_reg_co, fe_reg_pr )
    return(fe_reg)