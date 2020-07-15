import pandas as pd
import numpy as np
import matplotlib as plt
from linearmodels import PanelOLS
import statsmodels.api as sm
import econtools as econ
import econtools.metrics as mt
import math
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

def table5_setting(data):
    df = data
    df = df[((df['turin_co_sample']==1) | (df['turin_pr_sample']==1)) & ((df['post_experience']>=5)|(df['post_experience'].isnull()==True))  & ((df['pre_experience']>=5)|(df['pre_experience'].isnull()==True))& (df['missing']==0)]
    df = df[(df['ctrl_pop_turin_co_sample']==1) | (df['ctrl_pop_turin_pr_sample']==1) | (df['ctrl_exp_turin_co_sample']==1) | (df['ctrl_exp_turin_pr_sample']==1) | (df['ctrl_pop_exp_turin_co_sample']==1) | (df['ctrl_pop_exp_turin_pr_sample']==1)]
    df = df.reset_index()

    #re-construct trend-pa
    id_auth_remained = df['id_auth'].unique()
    id_auth_remained_df = pd.DataFrame({'id_auth': [], 'group_num': []})
    for i in range(len(id_auth_remained)):
        id_auth_remained_df.loc[i,'id_auth'] = id_auth_remained[i]
        id_auth_remained_df.loc[i,'group_num'] = i+1

    for i in range(len(df)):
        for j in range(len(id_auth_remained_df)):
            if df.loc[i, 'id_auth'] == id_auth_remained_df.loc[j, 'id_auth']:
                df.loc[i, 'id_auth_remained'] = j+1
    
    id_auth_remained_dum = pd.get_dummies(df['id_auth_remained']).rename(columns=lambda x: 'id_auth_remained' + str(x))
    df = pd.concat([df, id_auth_remained_dum],axis = 1)

    #re-contstruc trend-pa 시작
    for i in range(len(id_auth_remained_dum.columns)):
        df['trend_pa_remained_'+str(i+1)] = 0
        for j in range(len(df)):
            if df.loc[j, id_auth_remained_dum.columns[i]]==1 and df.loc[j, 'authority_code']!=3090272 and df.loc[j, 'authority_code']!=3070001:
                df.loc[j,'trend_pa_remained_'+str(i+1)] = 1
        df.drop([id_auth_remained_dum.columns[i]],axis = 1)

    #o = 'discount'
    #outcomes =[ 'overrun_ratio', 'days_to_award'] # ,
    #treatment = ['turin_co_sample','turin_pr_sample']
    return(df)

def table5_PanelA_odd(data, o):
    t = 'turin_co_sample'
    g = 'ctrl_exp'
    c_outcomes=1
    i = 5
    df1 = df
    df1 = df1[(df1[t]==1) & (df1[g +'_' + t]==1) & (df1['post_experience']>=i) & (df1['pre_experience']>=i)& (df1['post_experience'].isnull()==False) & (df1['pre_experience'].isnull()==False) & (df1['missing']==0) & (df1[o].isnull()==False) & (df1['fiscal_efficiency'].isnull()==False) & (df1['reserve_price'].isnull()==False)&(df1['municipality'].isnull()==False)]

    df1 = df1.reset_index() #to use loc
    df1 = df1.sort_values(by = 'authority_code', ascending = True)
    #df1 value checked 1262

    df1['ind'] = np.nan
    for i in range(len(df1)):
        if i == 0:
            df1.loc[i, 'ind'] = 1
        else:
            if df1.loc[i, 'authority_code'] != df1.loc[i-1, 'authority_code']:
                df1.loc[i, 'ind'] = 1

    #create dummies for administration-year pairs 
    all_years = df1['year'].unique()
    all_authorities = df1['authority_code'].unique()
    auth_year_reg_col = []
    for auth in all_authorities:
        for yr in all_years:
            df1['auth_year_' + str(auth)+'_' + str(yr)] = 0
            auth_year_reg_col.append('auth_year_' + str(auth)+'_' + str(yr))
            df1.loc[(df1['year']==yr) & (df1['authority_code']==auth), 'auth_year_' + str(auth)+'_' + str(yr) ] = 1

    ##regression for first stage
    #create dummies for work category
    all_categories = df1['work_category'].unique()
    for cat in all_categories:
        df1['cat_'+cat] = 0
        df1.loc[df1['work_category']==cat, 'cat_'+cat] =1

    ### Regression first stage 
    #setting
    work_dum = pd.get_dummies(df1['work_category']).rename(columns=lambda x: 'work_dum_' + str(x))
    year_dum = pd.get_dummies(df1['year']).rename(columns=lambda x: 'year_dum_' + str(x))
    auth_dum = pd.get_dummies(df1['authority_code']).rename(columns=lambda x: 'auth_dum_' + str(x))
    dum_df = pd.concat([work_dum, year_dum, auth_dum],axis = 1)
    #이렇게 해주고 부터 fe_reg_1 singular matrix 걸림
    df1 = pd.concat([df1,dum_df],axis = 1)

    work_list = list(work_dum.columns)
    year_list = list(year_dum.columns)
    auth_list = list(auth_dum.columns)

    reg_col = []
    for i in work_list:
        reg_col.append(i)
    for j in year_list:
        reg_col.append(j)
    for k in auth_list:
        reg_col.append(k)

    exog_var = ['fpsb_auction','reserve_price','municipality','fiscal_efficiency']
    exog = exog_var + reg_col 
    exog.remove('year_dum_2000.0')

    exog.remove('work_dum_OG01')
    exog.remove('auth_dum_3.0')
    exog.remove('auth_dum_1708.0')

    #reg_col for co_sample, discount,ctrl_exp, fe_reg_1&2
    #값은 다음
    #exog = exog_var + reg_col

    #1. reg
    fe_reg_1 = mt.reg(df1, o, exog, cluster = 'auth_anno', addcons= True, check_colinear = True)
    
    #2. reg
    fe_reg_2 = mt.reg(df1, o, exog, cluster = 'authority_code',addcons= True, check_colinear = True)
    
    ci_1 = round(fe_reg_1.summary.loc['fpsb_auction',['CI_low', 'CI_high']])
    ci_2 = round(fe_reg_2.summary.loc['fpsb_auction',['CI_low', 'CI_high']])
    
    return(ci_1, ci_2)

def table5_PanelB_odd(data, o):
    t = 'turin_pr_sample'
    g = 'ctrl_exp'
    c_outcomes=1
    i = 5
    df1 = df
    df1 = df1[(df1[t]==1) & (df1[g +'_' + t]==1) & (df1['post_experience']>=i) & (df1['pre_experience']>=i)& (df1['post_experience'].isnull()==False) & (df1['pre_experience'].isnull()==False) & (df1['missing']==0) & (df1[o].isnull()==False) & (df1['fiscal_efficiency'].isnull()==False) & (df1['reserve_price'].isnull()==False)&(df1['municipality'].isnull()==False)]

    df1 = df1.reset_index() #to use loc
    df1 = df1.sort_values(by = 'authority_code', ascending = True)
    #df1 value checked 1262

    df1['ind'] = np.nan
    for i in range(len(df1)):
        if i == 0:
            df1.loc[i, 'ind'] = 1
        else:
            if df1.loc[i, 'authority_code'] != df1.loc[i-1, 'authority_code']:
                df1.loc[i, 'ind'] = 1

    #create dummies for administration-year pairs 
    all_years = df1['year'].unique()
    all_authorities = df1['authority_code'].unique()
    auth_year_reg_col = []
    for auth in all_authorities:
        for yr in all_years:
            df1['auth_year_' + str(auth)+'_' + str(yr)] = 0
            auth_year_reg_col.append('auth_year_' + str(auth)+'_' + str(yr))
            df1.loc[(df1['year']==yr) & (df1['authority_code']==auth), 'auth_year_' + str(auth)+'_' + str(yr) ] = 1

    ##regression for first stage
    #create dummies for work category
    all_categories = df1['work_category'].unique()
    for cat in all_categories:
        df1['cat_'+cat] = 0
        df1.loc[df1['work_category']==cat, 'cat_'+cat] =1

    ### Regression first stage 
    #setting
    work_dum = pd.get_dummies(df1['work_category']).rename(columns=lambda x: 'work_dum_' + str(x))
    year_dum = pd.get_dummies(df1['year']).rename(columns=lambda x: 'year_dum_' + str(x))
    auth_dum = pd.get_dummies(df1['authority_code']).rename(columns=lambda x: 'auth_dum_' + str(x))
    dum_df = pd.concat([work_dum, year_dum, auth_dum],axis = 1)
    #이렇게 해주고 부터 fe_reg_1 singular matrix 걸림
    df1 = pd.concat([df1,dum_df],axis = 1)

    work_list = list(work_dum.columns)
    year_list = list(year_dum.columns)
    auth_list = list(auth_dum.columns)

    reg_col = []
    for i in work_list:
        reg_col.append(i)
    for j in year_list:
        reg_col.append(j)
    for k in auth_list:
        reg_col.append(k)

    exog_var = ['fpsb_auction','reserve_price','municipality','fiscal_efficiency']
    exog = exog_var + reg_col 
    exog.remove('year_dum_2000.0')

    exog.remove('work_dum_OG01')
    exog.remove('auth_dum_3.0')
    exog.remove('auth_dum_1708.0')

    #reg_col for co_sample, discount,ctrl_exp, fe_reg_1&2
    #값은 다음
    #exog = exog_var + reg_col

    #1. reg
    fe_reg_1 = mt.reg(df1, o, exog, cluster = 'auth_anno', addcons= True, check_colinear = True)
    
    #2. reg
    fe_reg_2 = mt.reg(df1, o, exog, cluster = 'authority_code',addcons= True, check_colinear = True)
    
    ci_1 = round(fe_reg_1.summary.loc['fpsb_auction',['CI_low', 'CI_high']])
    ci_2 = round(fe_reg_2.summary.loc['fpsb_auction',['CI_low', 'CI_high']])
    
    return(ci_1, ci_2)


def table5_PanelA_even(data, o):
    t = 'turin_co_sample'
    g = 'ctrl_exp'
    c_outcomes=1
    i = 5
    df1 = df
    df1 = df1[(df1[t]==1) & (df1[g +'_' + t]==1) & (df1['post_experience']>=i) & (df1['pre_experience']>=i)& (df1['post_experience'].isnull()==False) & (df1['pre_experience'].isnull()==False) & (df1['missing']==0) & (df1[o].isnull()==False) & (df1['fiscal_efficiency'].isnull()==False) & (df1['reserve_price'].isnull()==False)&(df1['municipality'].isnull()==False)&(df1['trend'].isnull()==False)&(df1['trend_treat'].isnull()==False)]

    df1 = df1.reset_index() #to use loc
    df1 = df1.sort_values(by = 'authority_code', ascending = True)
    #df1 value checked 1262

    df1['ind'] = np.nan
    for i in range(len(df1)):
        if i == 0:
            df1.loc[i, 'ind'] = 1
        else:
            if df1.loc[i, 'authority_code'] != df1.loc[i-1, 'authority_code']:
                df1.loc[i, 'ind'] = 1

    #create dummies for administration-year pairs 
    all_years = df1['year'].unique()
    all_authorities = df1['authority_code'].unique()
    auth_year_reg_col = []
    for auth in all_authorities:
        for yr in all_years:
            df1['auth_year_' + str(auth)+'_' + str(yr)] = 0
            auth_year_reg_col.append('auth_year_' + str(auth)+'_' + str(yr))
            df1.loc[(df1['year']==yr) & (df1['authority_code']==auth), 'auth_year_' + str(auth)+'_' + str(yr) ] = 1

    ##regression for first stage
    #create dummies for work category
    all_categories = df1['work_category'].unique()
    for cat in all_categories:
        df1['cat_'+cat] = 0
        df1.loc[df1['work_category']==cat, 'cat_'+cat] =1

    ### Regression first stage 
    #setting
    work_dum = pd.get_dummies(df1['work_category']).rename(columns=lambda x: 'work_dum_' + str(x))
    year_dum = pd.get_dummies(df1['year']).rename(columns=lambda x: 'year_dum_' + str(x))
    auth_dum = pd.get_dummies(df1['authority_code']).rename(columns=lambda x: 'auth_dum_' + str(x))
    
    dum_df = pd.concat([work_dum, year_dum, auth_dum],axis = 1)
    #이렇게 해주고 부터 fe_reg_1 singular matrix 걸림
    df1 = pd.concat([df1,dum_df],axis = 1)

    work_list = list(work_dum.columns)
    year_list = list(year_dum.columns)
    auth_list = list(auth_dum.columns)

    reg_col = []
    for i in work_list:
        reg_col.append(i)
    for j in year_list:
        reg_col.append(j)
    #for k in auth_list:
    #    reg_col.append(k)

    exog_var = ['fpsb_auction','reserve_price','municipality','fiscal_efficiency','trend','trend_treat']
    
    for i in range(1,36):
        exog_var.append('trend_pa_remained_'+str(i))
        
    exog = exog_var + reg_col 
    

    exog.remove('year_dum_2000.0')
    exog.remove('work_dum_OG01')
    #exog.remove('auth_dum_3.0')
    #exog.remove('auth_dum_1246.0')
    for i in [2,4,6,7,9,11,12,13,15,16,17,18,20,21,22,23,24,25,26,28,34,35]:
        exog.remove('trend_pa_remained_'+str(i))

    #1. reg
    fe_reg_1 = mt.reg(df1, o, exog, cluster = 'auth_anno', check_colinear = True)

    #2. reg
    fe_reg_2 = mt.reg(df1, o, exog, cluster = 'authority_code', check_colinear = True)
    
    ci_1 = round(fe_reg_1.summary.loc['fpsb_auction',['CI_low', 'CI_high']])
    ci_2 = round(fe_reg_2.summary.loc['fpsb_auction',['CI_low', 'CI_high']])
    
    return(ci_1, ci_2)

def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

def table5_PanelB_even(data, o):
    
    t = 'turin_pr_sample'
    g = 'ctrl_exp'
    c_outcomes=1
    i = 5
    df1 = df
    df1 = df1[(df1[t]==1) & (df1[g +'_' + t]==1) & (df1['post_experience']>=i) & (df1['pre_experience']>=i)& (df1['post_experience'].isnull()==False) & (df1['pre_experience'].isnull()==False) & (df1['missing']==0) & (df1[o].isnull()==False) & (df1['fiscal_efficiency'].isnull()==False) & (df1['reserve_price'].isnull()==False)&(df1['municipality'].isnull()==False)&(df1['trend'].isnull()==False)&(df1['trend_treat'].isnull()==False)]

    df1 = df1.reset_index() #to use loc
    df1 = df1.sort_values(by = 'authority_code', ascending = True)
    #df1 value checked 1262

    df1['ind'] = np.nan
    for i in range(len(df1)):
        if i == 0:
            df1.loc[i, 'ind'] = 1
        else:
            if df1.loc[i, 'authority_code'] != df1.loc[i-1, 'authority_code']:
                df1.loc[i, 'ind'] = 1

    #create dummies for administration-year pairs 
    all_years = df1['year'].unique()
    all_authorities = df1['authority_code'].unique()
    auth_year_reg_col = []
    for auth in all_authorities:
        for yr in all_years:
            df1['auth_year_' + str(auth)+'_' + str(yr)] = 0
            auth_year_reg_col.append('auth_year_' + str(auth)+'_' + str(yr))
            df1.loc[(df1['year']==yr) & (df1['authority_code']==auth), 'auth_year_' + str(auth)+'_' + str(yr) ] = 1

    ##regression for first stage
    #create dummies for work category
    all_categories = df1['work_category'].unique()
    for cat in all_categories:
        df1['cat_'+cat] = 0
        df1.loc[df1['work_category']==cat, 'cat_'+cat] =1

    ### Regression first stage 
    #setting
    work_dum = pd.get_dummies(df1['work_category']).rename(columns=lambda x: 'work_dum_' + str(x))
    year_dum = pd.get_dummies(df1['year']).rename(columns=lambda x: 'year_dum_' + str(x))
    auth_dum = pd.get_dummies(df1['authority_code']).rename(columns=lambda x: 'auth_dum_' + str(x))
    
    dum_df = pd.concat([work_dum, year_dum, auth_dum],axis = 1)
    df1 = pd.concat([df1,dum_df],axis = 1)

    work_list = list(work_dum.columns)
    year_list = list(year_dum.columns)
    auth_list = list(auth_dum.columns)

    reg_col = []
    for i in work_list:
        reg_col.append(i)
    for j in year_list:
        reg_col.append(j)
    #for k in auth_list:
    #    reg_col.append(k)

    exog_var = ['fpsb_auction','reserve_price','municipality','fiscal_efficiency','trend','trend_treat']
    
    for i in range(1,36):
        exog_var.append('trend_pa_remained_'+str(i))
        
    exog = exog_var + reg_col 
    

    exog.remove('year_dum_2000.0')
    exog.remove('work_dum_OG01')
    exog.remove('year_dum_2006.0')
    #exog.remove('auth_dum_3.0')
    #exog.remove('auth_dum_1246.0')
    for i in [2,4,6,7,9,11,12,13,15,16,17,18,20,21,22,23,24,25,26,28,34,35]:
        exog.remove('trend_pa_remained_'+str(i))
    

    #1. reg
    fe_reg_1 = mt.reg(df1, o, exog, cluster = 'auth_anno', check_colinear = True)

    #2. reg
    
    fe_reg_2 = mt.reg(df1, o, exog, cluster = 'authority_code', check_colinear = True)
    
    ci_1 = fe_reg_1.summary.loc['fpsb_auction',['CI_low', 'CI_high']]
    ci_2 = fe_reg_2.summary.loc['fpsb_auction',['CI_low', 'CI_high']]
    
    return(ci_1,ci_2)