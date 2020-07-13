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


def table4_setting(data):
    
    df = data
    auth_list = df['authority_code'].unique()
    
    for i in range(len(auth_list)):
        name = 'trend_pa_'+str(i+1)
        df[name] = 0
        df.loc[(df['authority_code']==3090272)| (df['authority_code']==3070001), name]
        for j in range(len(df)):
            if df.loc[j, 'id_auth'] == i+1 :
                df.loc[j, name] = df.loc[j,'trend']

    return(df)


def table4_PanelA_odd(data):
    df = data
    outcome = ['discount', 'delay_ratio', 'overrun_ratio', 'days_to_award']
    t = 'turin_co_sample'
    work_list = df['work_category'].unique()
    year_list = df['year'].unique()
    for o in outcome:
        df_reg_t = df[(df[t]==1)&(df['ctrl_exp_' + t]==1)&(df['post_experience']>= 5) & (df['pre_experience']>=5) &(df['post_experience'].isnull() == False ) & (df['pre_experience'].isnull()==False)&
                      (df['missing']==0)]
        idx = df_reg_t[df_reg_t[o].isnull()==True].index
        df_reg_o = df_reg_t.drop(idx)

        #first, make a column list
        reg_col = []
        for i in work_list:
            reg_col.append(i)
        for j in year_list:
            reg_col.append(j)
        exog_var = ['fpsb_auction','reserve_price','municipality','fiscal_efficiency']
        exog = exog_var + reg_col 


        #check multicollinearity
        X = df_reg_o.loc[:,exog]
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


        if o == 'discount':
            fe_reg_discount = mt.reg(df_reg_o, o, exog, fe_name = 'authority_code', cluster = 'auth_anno')
        elif o == 'delay_ratio':
            fe_reg_delay = mt.reg(df_reg_o, o, exog, fe_name = 'authority_code', cluster = 'auth_anno')
        elif o == 'overrun_ratio':
            fe_reg_overrun = mt.reg(df_reg_o, o, exog, fe_name = 'authority_code', cluster = 'auth_anno',check_colinear = True)
        else :
            fe_reg_award = mt.reg(df_reg_o, o, exog, fe_name = 'authority_code', cluster = 'auth_anno',check_colinear = True)

    fe_reg = (fe_reg_discount, fe_reg_delay, fe_reg_overrun, fe_reg_award )
    return(fe_reg)


def table4_PanelA_even(data):
    df = data
    outcome = ['discount', 'delay_ratio', 'overrun_ratio', 'days_to_award']
    t = 'turin_co_sample'
    work_list = df['work_category'].unique()
    year_list = df['year'].unique()
    outcome = ['discount', 'delay_ratio','overrun_ratio', 'days_to_award']

    for o in outcome:
        df_reg_co = df[(df['turin_co_sample']==1)&(df['ctrl_exp_turin_co_sample']==1)&(df['post_experience']>= 5) & (df['pre_experience']>=5) &(df['post_experience'].isnull() == False ) & 
                       (df['pre_experience'].isnull()==False)&
                       (df['missing']==0)]

        #first, make a column list
        reg_col = ['trend_pa_2','trend_pa_3','trend_pa_5','trend_pa_8','trend_pa_15','trend_pa_19','trend_pa_24','trend_pa_29','trend_pa_1231',
                  'trend_pa_1690','trend_pa_1721','trend_pa_1749','trend_pa_1839']
        for i in work_list:
            reg_col.append(i)
        for j in year_list:
            reg_col.append(j)
        exog_var = ['fpsb_auction','reserve_price','municipality','trend','trend_treat']
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

        if o == 'discount':
            fe_reg_discount = mt.reg(df_reg_co, o, exog, fe_name = 'authority_code', cluster = 'auth_anno')
        
        elif o == 'overrun_ratio':
            exog.remove('OS07')
            fe_reg_overrun = mt.reg(df_reg_co, o , exog, fe_name = 'authority_code', cluster = 'auth_anno')

        elif o == 'delay_ratio':
            fe_reg_delay = mt.reg(df_reg_co, o, exog, fe_name = 'authority_code', cluster = 'auth_anno')
        
        else:
            exog.remove('OG04')
            exog.remove('OS05')
            exog.remove('OS07')
            exog.remove('OS11')
            exog.remove('OS26')
            fe_reg_award = mt.reg(df_reg_co, o , exog, fe_name = 'authority_code', cluster = 'auth_anno')

    fe_reg = (fe_reg_discount, fe_reg_delay, fe_reg_overrun, fe_reg_award )
    return(fe_reg)


def table4_PanelB_odd(data):
    df = data
    outcome = ['discount', 'delay_ratio', 'overrun_ratio', 'days_to_award']
    t = 'turin_pr_sample'
    work_list = df['work_category'].unique()
    year_list = df['year'].unique()
    for o in outcome:
        df_reg_t = df[(df[t]==1)&(df['ctrl_exp_' + t]==1)&(df['post_experience']>= 5) & (df['pre_experience']>=5) &(df['post_experience'].isnull() == False ) & (df['pre_experience'].isnull()==False)&
                      (df['missing']==0)]
        idx = df_reg_t[df_reg_t[o].isnull()==True].index
        df_reg_o = df_reg_t.drop(idx)

        #first, make a column list
        reg_col = []
        for i in work_list:
            reg_col.append(i)
        for j in year_list:
            reg_col.append(j)
        exog_var = ['fpsb_auction','reserve_price','municipality','fiscal_efficiency']
        exog = exog_var + reg_col 


        #check multicollinearity
        X = df_reg_o.loc[:,exog]
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


        if o == 'discount':
            fe_reg_discount = mt.reg(df_reg_o, o, exog, fe_name = 'authority_code', cluster = 'auth_anno')
        elif o == 'delay_ratio':
            fe_reg_delay = mt.reg(df_reg_o, o, exog, fe_name = 'authority_code', cluster = 'auth_anno')
        elif o == 'overrun_ratio':
            fe_reg_overrun = mt.reg(df_reg_o, o, exog, fe_name = 'authority_code', cluster = 'auth_anno',check_colinear = True)
        else :
            fe_reg_award = mt.reg(df_reg_o, o, exog, fe_name = 'authority_code', cluster = 'auth_anno',check_colinear = True)

    fe_reg = (fe_reg_discount, fe_reg_delay, fe_reg_overrun, fe_reg_award )
    return(fe_reg)


def table4_PanelB_even(data):
    df = data
    outcome = ['discount', 'delay_ratio', 'overrun_ratio', 'days_to_award']
    t = 'turin_pr_sample'
    work_list = df['work_category'].unique()
    year_list = df['year'].unique()
    
    for o in outcome:
        df_reg_co = df[(df['turin_pr_sample']==1)&(df['ctrl_exp_turin_pr_sample']==1)&(df['post_experience']>= 5) & (df['pre_experience']>=5) &(df['post_experience'].isnull() == False ) & 
                       (df['pre_experience'].isnull()==False)&
                       (df['missing']==0)]

        #first, make a column list
        reg_col = ['trend_pa_2','trend_pa_3','trend_pa_5','trend_pa_8','trend_pa_15','trend_pa_19','trend_pa_24','trend_pa_29','trend_pa_33','trend_pa_585','trend_pa_1231',
                  'trend_pa_1480', 'trend_pa_1690','trend_pa_1721','trend_pa_1749','trend_pa_1839']
        for i in work_list:
            reg_col.append(i)
        for j in year_list:
            reg_col.append(j)
        exog_var = ['fpsb_auction','reserve_price','municipality','trend','trend_treat']
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

        if o == 'discount':
            fe_reg_discount = mt.reg(df_reg_co, o , exog, fe_name = 'authority_code', cluster = 'auth_anno',addcons = True,check_colinear = True)
    
        elif o == 'delay_ratio':
            exog.remove('OG09')
            fe_reg_delay = mt.reg(df_reg_co, o , exog, fe_name = 'authority_code', cluster = 'auth_anno',addcons = True,check_colinear = True)

        elif o == 'overrun_ratio':
            exog.remove('OS07')#cost
            exog.remove('OS09')#cost
            fe_reg_overrun = mt.reg(df_reg_co, o , exog, fe_name = 'authority_code', cluster = 'auth_anno',addcons = True,check_colinear = True)

        elif o == 'days_to_award':
            exog.remove('OG04')
            exog.remove('OS05')
            exog.remove('OS11')
            exog.remove('OS26')
            fe_reg_award = mt.reg(df_reg_co, o , exog, fe_name = 'authority_code', cluster = 'auth_anno',addcons = True,check_colinear = True)
        
    fe_reg = (fe_reg_discount, fe_reg_delay, fe_reg_overrun, fe_reg_award )
    return(fe_reg)