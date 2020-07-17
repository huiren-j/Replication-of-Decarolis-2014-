import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
from auxiliary.table5 import *
from auxiliary.table6 import *
from auxiliary.table7 import *
from auxiliary.extension import *
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

    #re-construct trend-pa: setting
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

    #re-construct trend-pa
    for i in range(len(id_auth_remained_dum.columns)):
        df['trend_pa_remained_'+str(i+1)] = 0
        for j in range(len(df)):
            if df.loc[j, id_auth_remained_dum.columns[i]]==1 and df.loc[j, 'authority_code']!=3090272 and df.loc[j, 'authority_code']!=3070001:
                df.loc[j,'trend_pa_remained_'+str(i+1)] = 1
        df.drop([id_auth_remained_dum.columns[i]],axis = 1)

    return(df)

def table5_PanelA_odd(data):
    outcomes = ['discount','delay_ratio','overrun_ratio','days_to_award']
    t = 'turin_co_sample'
    g = 'ctrl_exp'
    c_outcomes=1
    i = 5
    df1 = data
    df1_tmp = df1[(df1[t]==1)& (df1[g +'_' + t]==1) & (df1['post_experience']>=i) & (df1['pre_experience']>=i)& (df1['post_experience'].isnull()==False) & (df1['pre_experience'].isnull()==False) & (df1['missing']==0) & (df1['fiscal_efficiency'].isnull()==False) & (df1['reserve_price'].isnull()==False)&(df1['municipality'].isnull()==False)]
    for o in outcomes:
        df1 =  df1_tmp[df1_tmp[o].isnull()==False] 
        df1 = df1.reset_index() 
        df1 = df1.sort_values(by = 'authority_code', ascending = True)

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
        for k in auth_list:
            reg_col.append(k)

        exog_var = ['fpsb_auction','reserve_price','municipality','fiscal_efficiency']
        exog = exog_var + reg_col 
        exog.remove('year_dum_2000.0')

        exog.remove('work_dum_OG01')
        exog.remove('auth_dum_3.0')
        exog.remove('auth_dum_1708.0')

        #1. reg
        fe_reg_1 = mt.reg(df1, o, exog, cluster = 'auth_anno', addcons= True, check_colinear = True)
        #2. reg
        fe_reg_2 = mt.reg(df1, o, exog, cluster = 'authority_code',addcons= True, check_colinear = True)

        ci_1 = fe_reg_1.summary.loc['fpsb_auction',['CI_low', 'CI_high']].round()
        ci_2 = fe_reg_2.summary.loc['fpsb_auction',['CI_low', 'CI_high']].round()

        if o == 'discount':
            ci_discount = pd.DataFrame((ci_1,ci_2))
        elif o == 'delay_ratio':
            ci_delay_ratio = pd.DataFrame((ci_1,ci_2))
        elif o == 'overrun_ratio':
            ci_overrun_ratio = pd.DataFrame((ci_1,ci_2))
        else:
            ci_days_to_award = pd.DataFrame((ci_1,ci_2))

    ci = pd.concat([ci_discount,ci_delay_ratio,ci_overrun_ratio,ci_days_to_award],axis=1).reset_index()
    del ci['index']
    return(ci)

def table5_PanelA_even(data):
    outcomes = ['discount','delay_ratio','overrun_ratio','days_to_award']
    t = 'turin_co_sample'
    g = 'ctrl_exp'
    c_outcomes=1
    i = 5
    df1 = data
    df1_tmp = df1[(df1[t]==1)& (df1[g +'_' + t]==1) & (df1['post_experience']>=i) & (df1['pre_experience']>=i)& (df1['post_experience'].isnull()==False) & (df1['pre_experience'].isnull()==False) & (df1['missing']==0) & (df1['fiscal_efficiency'].isnull()==False) & (df1['reserve_price'].isnull()==False)&(df1['municipality'].isnull()==False)]
    for o in outcomes:
        df1 =  df1_tmp[df1_tmp[o].isnull()==False] 
        df1 = df1.reset_index()
        df1 = df1.sort_values(by = 'authority_code', ascending = True)

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


        exog_var = ['fpsb_auction','reserve_price','municipality','fiscal_efficiency','trend','trend_treat']

        for i in range(1,36):
            exog_var.append('trend_pa_remained_'+str(i))

        exog = exog_var + reg_col 


        exog.remove('year_dum_2000.0')
        exog.remove('work_dum_OG01')
        for i in [2,4,6,7,9,11,12,13,15,16,17,18,20,21,22,23,24,25,26,28,34,35]:
            exog.remove('trend_pa_remained_'+str(i))

        #1. reg
        fe_reg_1 = mt.reg(df1, o, exog, cluster = 'auth_anno', check_colinear = True)
    
        #2. reg
        fe_reg_2 = mt.reg(df1, o, exog, cluster = 'authority_code', check_colinear = True)

        
        ci_1 = fe_reg_1.summary.loc['fpsb_auction',['CI_low', 'CI_high']].round()
        ci_2 = fe_reg_2.summary.loc['fpsb_auction',['CI_low', 'CI_high']].round()

        if o == 'discount':
            ci_discount = pd.DataFrame((ci_1,ci_2))
        elif o == 'delay_ratio':
            ci_delay_ratio = pd.DataFrame((ci_1,ci_2))
        elif o == 'overrun_ratio':
            ci_overrun_ratio = pd.DataFrame((ci_1,ci_2))
        else:
            ci_days_to_award = pd.DataFrame((ci_1,ci_2))

    ci = pd.concat([ci_discount,ci_delay_ratio,ci_overrun_ratio,ci_days_to_award],axis=1).reset_index()
    del ci['index']
    return(ci)

def table5_PanelB_odd(data):
    outcomes = ['discount','delay_ratio','overrun_ratio','days_to_award']
    t = 'turin_pr_sample'
    g = 'ctrl_exp'
    c_outcomes=1
    i = 5
    df1 = data
    df1_tmp = df1[(df1[t]==1)& (df1[g +'_' + t]==1) & (df1['post_experience']>=i) & (df1['pre_experience']>=i)& (df1['post_experience'].isnull()==False) & (df1['pre_experience'].isnull()==False) & (df1['missing']==0) & (df1['fiscal_efficiency'].isnull()==False) & (df1['reserve_price'].isnull()==False)&(df1['municipality'].isnull()==False)]
    for o in outcomes:
        df1 =  df1_tmp[df1_tmp[o].isnull()==False] 
        df1 = df1.reset_index()
        df1 = df1.sort_values(by = 'authority_code', ascending = True)
        
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
        for k in auth_list:
            reg_col.append(k)

        exog_var = ['fpsb_auction','reserve_price','municipality','fiscal_efficiency']
        exog = exog_var + reg_col 
        exog.remove('year_dum_2000.0')

        exog.remove('work_dum_OG01')
        exog.remove('auth_dum_3.0')
        exog.remove('auth_dum_1708.0')


        #1. reg
        fe_reg_1 = mt.reg(df1, o, exog, cluster = 'auth_anno', addcons= True, check_colinear = True)
        #2. reg
        fe_reg_2 = mt.reg(df1, o, exog, cluster = 'authority_code',addcons= True, check_colinear = True)


        ci_1 = fe_reg_1.summary.loc['fpsb_auction',['CI_low', 'CI_high']].round()
        ci_2 = fe_reg_2.summary.loc['fpsb_auction',['CI_low', 'CI_high']].round()

        if o == 'discount':
            ci_discount = pd.DataFrame((ci_1,ci_2))
        elif o == 'delay_ratio':
            ci_delay_ratio = pd.DataFrame((ci_1,ci_2))
        elif o == 'overrun_ratio':
            ci_overrun_ratio = pd.DataFrame((ci_1,ci_2))
        else:
            ci_days_to_award = pd.DataFrame((ci_1,ci_2))

    ci = pd.concat([ci_discount,ci_delay_ratio,ci_overrun_ratio,ci_days_to_award],axis=1).reset_index()
    del ci['index']
    return(ci)


def table5_PanelB_even(data):
    outcomes = ['discount','delay_ratio','overrun_ratio','days_to_award']
    t = 'turin_pr_sample'
    g = 'ctrl_exp'
    c_outcomes=1
    i = 5
    df1 = data
    df1_tmp = df1[(df1[t]==1)& (df1[g +'_' + t]==1) & (df1['post_experience']>=i) & (df1['pre_experience']>=i)& (df1['post_experience'].isnull()==False) & (df1['pre_experience'].isnull()==False) & (df1['missing']==0) & (df1['fiscal_efficiency'].isnull()==False) & (df1['reserve_price'].isnull()==False)&(df1['municipality'].isnull()==False)]
    for o in outcomes:
        df1 =  df1_tmp[df1_tmp[o].isnull()==False] 
        df1 = df1.reset_index()
        df1 = df1.sort_values(by = 'authority_code', ascending = True)
        
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


        exog_var = ['fpsb_auction','reserve_price','municipality','fiscal_efficiency','trend','trend_treat']

        for i in range(1,36):
            exog_var.append('trend_pa_remained_'+str(i))

        exog = exog_var + reg_col 

        exog.remove('year_dum_2000.0')
        exog.remove('year_dum_2006.0')
        exog.remove('work_dum_OG01')
        for i in [2,4,6,7,9,11,12,13,15,16,17,18,20,21,22,23,24,25,26,28,34,35]:
            exog.remove('trend_pa_remained_'+str(i))
        

        #1. reg
        fe_reg_1 = mt.reg(df1, o, exog, cluster = 'auth_anno', check_colinear = True)
    
        #2. reg
        fe_reg_2 = mt.reg(df1, o, exog, cluster = 'authority_code', check_colinear = True)


        ci_1 = fe_reg_1.summary.loc['fpsb_auction',['CI_low', 'CI_high']].round()
        ci_2 = fe_reg_2.summary.loc['fpsb_auction',['CI_low', 'CI_high']].round()

        if o == 'discount':
            ci_discount = pd.DataFrame((ci_1,ci_2))
        elif o == 'delay_ratio':
            ci_delay_ratio = pd.DataFrame((ci_1,ci_2))
        elif o == 'overrun_ratio':
            ci_overrun_ratio = pd.DataFrame((ci_1,ci_2))
        else:
            ci_days_to_award = pd.DataFrame((ci_1,ci_2))

    ci = pd.concat([ci_discount,ci_delay_ratio,ci_overrun_ratio,ci_days_to_award],axis=1).reset_index()
    del ci['index']
    return(ci)



def table5_PanelA_odd_row3(data):
    outcomes = ['discount','delay_ratio','overrun_ratio'] #Aodd_days to award값 안나옴
    t = 'turin_co_sample'
    g = 'ctrl_exp'
    c_outcomes=1
    i = 5
    df1 = data
    df1_tmp = df1[(df1[t]==1)& (df1[g +'_' + t]==1) & (df1['post_experience']>=i) & (df1['pre_experience']>=i)& (df1['post_experience'].isnull()==False) & (df1['pre_experience'].isnull()==False) & (df1['missing']==0) & (df1['fiscal_efficiency'].isnull()==False) & (df1['reserve_price'].isnull()==False)&(df1['municipality'].isnull()==False)]
    for o in outcomes:
        df1 =  df1_tmp[df1_tmp[o].isnull()==False] 
        df1 = df1.reset_index() 
        df1 = df1.sort_values(by = 'authority_code', ascending = True)

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
        for k in auth_list:
            reg_col.append(k)

        exog_var = ['fpsb_auction','reserve_price','municipality','fiscal_efficiency']
        exog = exog_var + reg_col 
        exog.remove('year_dum_2000.0')

        exog.remove('work_dum_OG01')
        exog.remove('auth_dum_3.0')
        exog.remove('auth_dum_1708.0')

        #1. reg
        fe_reg_1 = mt.reg(df1, o, exog, cluster = 'auth_anno', addcons= True, check_colinear = True)
        #2. reg
        fe_reg_2 = mt.reg(df1, o, exog, cluster = 'authority_code',addcons= True, check_colinear = True)


        ci_1 = fe_reg_1.summary.loc['fpsb_auction',['CI_low', 'CI_high']].round()
        ci_2 = fe_reg_2.summary.loc['fpsb_auction',['CI_low', 'CI_high']].round()

        if o == 'discount':
            ci_discount = pd.DataFrame((ci_1,ci_2))
        elif o == 'delay_ratio':
            ci_delay_ratio = pd.DataFrame((ci_1,ci_2))
        elif o == 'overrun_ratio':
            ci_overrun_ratio = pd.DataFrame((ci_1,ci_2))
        else:
            ci_days_to_award = pd.DataFrame((ci_1,ci_2))

        reg_col = auth_year_reg_col
        for cat in all_categories:
            reg_col.append('cat_'+cat)
        exog_var = ['reserve_price','municipality','fiscal_efficiency']
        exog = exog_var + reg_col


        if o != 'overrun_ratio':
            exog.remove('auth_year_4.0_2000.0')
            exog.remove('auth_year_6.0_2000.0') 
            exog.remove('auth_year_16.0_2002.0')
            exog.remove('auth_year_16.0_2003.0')
            exog.remove('auth_year_16.0_2004.0')
            exog.remove('auth_year_1246.0_2000.0')
            exog.remove('cat_OS07')
            exog.remove('fiscal_efficiency')

        else:
            exog.remove('auth_year_4.0_2000.0')
            exog.remove('auth_year_6.0_2000.0') 
            exog.remove('auth_year_16.0_2002.0')
            exog.remove('auth_year_16.0_2003.0')
            exog.remove('auth_year_16.0_2004.0')
            exog.remove('auth_year_1246.0_2000.0')
            exog.remove('fiscal_efficiency')
            exog.remove('auth_year_6.0_2005.0')
            exog.remove('auth_year_9.0_2006.0')
            exog.remove('auth_year_25.0_2002.0')

        if o != 'discount':
            exog.remove('auth_year_20.0_2001.0')
            exog.remove('auth_year_20.0_2003.0')

        fe_reg_3 = mt.reg(df1, o, exog, cluster = 'auth_anno', check_colinear = True)

        df1['dummy_cat'] = 0
    
        beta_cat_list = []
        beta_list = []
        for i in range(len(exog)):
            for cat in all_categories:
                if exog[i] == 'cat_'+cat:
                    beta_cat_list.append(exog[i])
            for exo in exog_var:
                if exog[i] == exo:
                    beta_list.append(exog[i])

        if o == 'discount':
            discount_hat = fe_reg_3.yhat
            for i in range(len(df1)):
                for cat in beta_cat_list:
                    df1.loc[i, 'discount_beta'] =  df1.loc[i,'dummy_cat']-df1.loc[i,cat] * fe_reg_3.beta[cat]
                    for exo in beta_list:
                        df1.loc[i,'discount_beta'] = df1.loc[i,'discount_beta']- df1.loc[i,exo]*fe_reg_3.beta[exo]

            df1['discount_beta'] = discount_hat - df1['discount_beta']

        elif o == 'delay_ratio':
            delay_ratio_hat = fe_reg_3.yhat
            for i in range(len(df1)):
                for cat in beta_cat_list:
                    df1.loc[i, 'delay_ratio_beta'] =  df1.loc[i,'dummy_cat']-df1.loc[i,cat] * fe_reg_3.beta[cat]
                    for exo in beta_list:
                        df1.loc[i,'delay_ratio_beta'] = df1.loc[i,'delay_ratio_beta']- df1.loc[i,exo]*fe_reg_3.beta[exo]

            df1['delay_ratio_beta'] = delay_ratio_hat - df1['delay_ratio_beta']
        elif o == 'overrun_ratio':
            overrun_ratio_hat = fe_reg_3.yhat
            for i in range(len(df1)):
                for cat in beta_cat_list:
                    df1.loc[i, 'overrun_ratio_beta'] =  df1.loc[i,'dummy_cat']-df1.loc[i,cat] * fe_reg_3.beta[cat]
                    for exo in beta_list:
                        df1.loc[i,'overrun_ratio_beta'] = df1.loc[i,'overrun_ratio_beta']- df1.loc[i,exo]*fe_reg_3.beta[exo]

            df1['overrun_ratio_beta'] = overrun_ratio_hat - df1['overrun_ratio_beta']
        else:
            days_to_award_hat = fe_reg_3.yhat
            for i in range(len(df1)):
                for cat in beta_cat_list:
                    df1.loc[i, 'days_to_award_beta'] =  df1.loc[i,'dummy_cat']-df1.loc[i,cat] * fe_reg_3.beta[cat]
                    for exo in beta_list:
                        df1.loc[i,'days_to_award_beta'] = df1.loc[i,'days_to_award_beta']- df1.loc[i,exo]*fe_reg_3.beta[exo]

            df1['days_to_award_beta'] = days_to_award_hat - df1['days_to_award_beta']

        #create weigths - working well
        nrep_s = df1.groupby(['authority_code','year']).size().unstack(level=1)
        df1_nrep = pd.DataFrame(nrep_s)/len(df1)
        df1['weights'] = np.nan
        for auth in all_authorities:
            for yr in all_years:
                df1.loc[(df1['authority_code']==auth)&(df1['year']==yr),'weights'] = df1_nrep.loc[auth, yr]

        #Keep only beta coefficients for state*year terms
        collapse_list = [o +'_beta', 'authority_code', 'year', 'fpsb_auction', 'municipality', 'fiscal_efficiency', 'missing', 'turin_co_sample', 'weights'] + year_list + auth_list
        collapse = df1.groupby(['auth_anno'])[collapse_list].mean()
        df2 = collapse
        df2 = df2.reset_index()

        #Core conley-taber method
        exog_var = ['fpsb_auction', 'municipality', 'fiscal_efficiency']
        reg_col = []
        reg_col_new = []

        #reg_col.append(j)
        for i in auth_list:
            reg_col.append(i)
        for j in year_list:
            reg_col.append(j)

        for k in reg_col:
            for j in df2.columns:
                if k ==j:
                    reg_col_new.append(j)

        exog = exog_var + reg_col_new


        X = df2.loc[:,exog]
        vif = calc_vif(X)

        #delete from col list
        for i in range(len(vif)):
            if np.isnan(vif.loc[i, 'VIF']) == True:
                reg_col.remove(vif.loc[i, 'variables'])

        exog = exog_var + reg_col_new

        exog.remove('year_dum_2000.0')
        exog.remove('auth_dum_3.0')
        exog.remove('auth_dum_1866.0')

        wls = mt.reg(df2, o+'_beta', exog , cluster = 'auth_anno',addcons = True, awt_name = 'weights')

        #predic res
        df2['eta'] = wls.resid
        df2['eta'] = df2['eta']+ df2['fpsb_auction']*wls.beta['fpsb_auction']

        #Create tilde
        df2 = df2.sort_values(by = 'year',ascending = True)
        df2_wls= df2[(df2['authority_code']==3090272) | (df2['authority_code']==3070001)]
        df2_wls = pd.DataFrame(df2_wls.groupby(['year'])['fpsb_auction'].mean())
        for i in range(len(df2)):
            if df2.loc[i, 'authority_code']==3090272 or df2.loc[i, 'authority_code']==3070001:
                for j in list(df2_wls.index):
                    if df2.loc[i, 'year'] == j:
                        df2.loc[i,'djtga'] = df2_wls.loc[j, 'fpsb_auction']

        df2_wls = pd.DataFrame(df2.groupby(['year'])['djtga'].sum())
        for i in range(len(df2)):
            for j in list(df2_wls.index):
                if df2.loc[i, 'year'] == j:
                    df2.loc[i,'djt'] = df2_wls.loc[j, 'djtga']

        df2 = df2.sort_values(by = 'authority_code', ascending = True)
        df2_wls = pd.DataFrame(df2.groupby(['authority_code'])['djt'].mean())
        for i in range(len(df2)):
            for j in list(df2_wls.index):
                if df2.loc[i, 'authority_code'] == j:
                    df2.loc[i,'meandjt'] = df2_wls.loc[j, 'djt']

        df2['dtil'] = df2['djt'] - df2['meandjt']

        #obtain diff in diff coeff
        #renormalize weights
        df2.loc[(df2['authority_code']==3090272) | (df2['authority_code']==3070001),'tot_weights'] = df2['weights'].sum()
        df2['new_weights'] = df2['weights']/df2['tot_weights']
        df2_wls = df2[(df2['authority_code']==3090272) | (df2['authority_code']==3070001)]
        wls_2 = mt.reg(df2_wls, 'eta' , 'dtil' , awt_name = 'new_weights', addcons = True,check_colinear = True)

        alpha = [wls_2.beta['dtil']] 
        df2 = df2.drop(['tot_weights','new_weights'],axis = 1)

        #simulataneous for each public
        asim = []

        for auth in all_authorities:
            if auth !=3090272 and auth !=3070001:
                df2.loc[df2['authority_code']==auth, 'tot_weights'] = df2['weights'].sum()
                df2['new_weights'] = df2['weights']/df2['tot_weights']
                df2_wls_3 = df2[df2['authority_code']==auth]
                wls_3 = mt.reg(df2_wls_3, 'eta' , 'dtil' , awt_name = 'new_weights',check_colinear=True)
                asim.append(wls_3.beta['dtil'])
                df2 = df2.drop(['tot_weights','new_weights'],axis = 1)

        for i in range(len(asim)-1):
            alpha.append(alpha[0])

        asim_tmp = []
        for i in range(min(len(alpha),len(asim))):
            asim_tmp.append(alpha[i] - asim[i])

        #asim = asim_tmp
        df2['ci'] = np.nan
        df2['asim'] = np.nan
        for i in range(len(asim)):
            df2.loc[i, 'ci'] = asim_tmp[i]
            df2.loc[i, 'asim'] = asim[i]

        #form confidence level
        numst=len(asim)+1
        i025=math.floor(0.025*(numst-1))
        i025=max([i025,1])
        i975=math.ceil(0.975*(numst-1))
        i05=math.floor(0.050*(numst-1))
        i05=max([i05,i025+1])
        i95=math.ceil(0.950*(numst-1))
        i95=min([i95,numst-2])

        stima_ta = alpha[0]
        df2.sort_values(by = 'asim',ascending = True)
        ci_ta025 = min([df2.loc[i025,'ci'], df2.loc[i975, 'ci'] ])
        ci_ta975 = max([df2.loc[i025,'ci'], df2.loc[i975, 'ci'] ])

        if o == 'discount':
            discount_list = [round(ci_ta025), round(ci_ta975)]
        elif o == 'delay_ratio':
            delay_ratio_list = [ci_ta025, ci_ta975]
        elif o == 'overrun_ratio':
            overrun_ratio_list = [ci_ta025, ci_ta975]
        else:
            days_to_award_list = [ci_ta025, ci_ta975]
        
    final_list = discount_list + delay_ratio_list + overrun_ratio_list
    return(final_list)