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
from auxiliary.table7 import *
from auxiliary.table_formula import *

def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

'''extension with IE.dta'''

def extension_setting_IE(Authority, IE):
    df_IE = IE
    df_auth = Authority
    ctrl_exp_vlunt_list = df_IE.loc[df_IE['adopters_vlunt']==1,'experience'].unique()
    ctrl_pop_vlunt_list = df_IE.loc[df_IE['adopters_vlunt']==1,'population'].unique()
    df_IE['ctrl_exp_adopters_vlunt'] =0
    df_IE['ctrl_pop_adopters_vlunt'] =0
    for i in df_IE['experience'].unique():
        for j in range(len(ctrl_exp_vlunt_list)):
            if i == ctrl_exp_vlunt_list[j]:
                df_IE.loc[df_IE['experience']==i, 'ctrl_exp_adopters_vlunt'] =1
    for i in df_IE['population'].unique():
        for j in range(len(ctrl_pop_vlunt_list)):
            if i == ctrl_pop_vlunt_list[j]:
                df_IE.loc[df_IE['population']==i, 'ctrl_pop_adopters_vlunt'] =1
    
    work_dum = pd.get_dummies(df_IE['work_category']).rename(columns=lambda x: 'work_dum_' + str(x))
    year_dum = pd.get_dummies(df_IE['year']).rename(columns=lambda x: 'year_dum_' + str(x))
    df_IE = pd.concat([df_IE,work_dum],axis = 1)
    df_IE = pd.concat([df_IE,year_dum],axis = 1)

    return(df_IE)


def vlunt_col1(data):
    df= data
    df = df[(df['adopters_vlunt']==1)&(df['ctrl_exp_adopters_vlunt']==1)]
    
    year_list = df['year'].unique()
    outcome = ['discount', 'days_to_award']

    #col1 iyear
    for o in outcome:

        reg_col = []
        for j in year_list:
            reg_col.append('year_dum_' + str(j))

        exog_var = ['fpsb_auction']
        exog = exog_var + reg_col 


        #check multicollinearity
        X = df.loc[:,exog]
        vif = calc_vif(X)

        #delete from col list
        for i in range(len(vif)):
            if np.isnan(vif.loc[i, 'VIF']) == True:
                reg_col.remove(vif.loc[i, 'variables'])

        exog = exog_var + reg_col
        exog.remove('year_dum_2003.0')

        if o == 'discount':
            fe_reg_discount = mt.reg(df, o, exog, fe_name = 'authority_code', cluster = 'auth_anno', addcons = True, check_colinear = True)
        else :
            exog.remove('year_dum_2010.0')
            fe_reg_award = mt.reg(df, o, exog, fe_name = 'authority_code', cluster = 'auth_anno',addcons = True, check_colinear = True)

    fe_reg = (fe_reg_discount, fe_reg_award )
    return(fe_reg)


def vlunt_col2(data):
    df= data
    df = df[(df['adopters_vlunt']==1)&(df['ctrl_exp_adopters_vlunt']==1)]
    
    work_list = df['work_category'].unique()
    year_list = df['year'].unique()
    outcome = ['discount', 'days_to_award']
    
    #col2 iyear, iwork, reserve_price, municipality
    for o in outcome:
        reg_col = []
        for i in work_list:
            reg_col.append('work_dum_' + i)
        for j in year_list:
            reg_col.append('year_dum_' + str(j))

        exog_var = ['fpsb_auction','reserve_price','municipality']
        exog = exog_var + reg_col 

        #check multicollinearity

        #exog = exog_var + reg_col
        exog.remove('work_dum_OG03')
        exog.remove('municipality')
        exog.remove('year_dum_2003.0')

        if o == 'discount':
            fe_reg_discount = mt.reg(df, o, exog, fe_name = 'authority_code', cluster = 'auth_anno', addcons = True, check_colinear = True)
        else :
            exog.remove('year_dum_2010.0')
            fe_reg_award = mt.reg(df, o, exog, fe_name = 'authority_code', cluster = 'auth_anno', addcons = True, check_colinear = True)

    fe_reg = (fe_reg_discount, fe_reg_award )
    return(fe_reg)



'''extension with Authority.dta'''

def extension_setting_auth(Authority, IE):
    df_auth = Authority
    df_IE = IE
    
    ctrl_exp_forced_list = df_IE.loc[df_IE['forcedfp_strict'],'experience'].unique()
    ctrl_pop_forced_list = df_IE.loc[df_IE['forcedfp_strict'],'population'].unique()
    df_auth['ctrl_exp_forcedfp_co'] = df_auth['ctrl_exp_turin_co_sample']
    df_auth['ctrl_exp_forcedfp_pr'] = df_auth['ctrl_exp_turin_pr_sample']
    
    for i in ctrl_exp_forced_list:
        for j in range(len(df_auth)):
            if df_auth.loc[j, 'experience'] == i:
                df_auth.loc[j, 'ctrl_exp_forcedfp_co'] = 0
                df_auth.loc[j, 'ctrl_exp_forcedfp_pr'] = 0

    df_auth['ctrl_pop_forcedfp_co'] = df_auth['ctrl_pop_turin_co_sample']
    df_auth['ctrl_pop_forcedfp_pr'] = df_auth['ctrl_pop_turin_pr_sample']
    
    for i in ctrl_pop_forced_list:
        for j in range(len(df_auth)):
            if df_auth.loc[j, 'population'] == i:
                df_auth.loc[j, 'ctrl_pop_forcedfp_co'] = 0
                df_auth.loc[j, 'ctrl_pop_forcedfp_pr'] = 0
    
    return(df_auth)


def remove_aba_col1(data):
    #main table
    #col1
    df =data
    df = df[(df['turin_co_sample']==1)&(df['ctrl_exp_forcedfp_co']==1)&(df['post_experience']>= 5) & (df['pre_experience']>=5) &(df['post_experience'].isnull() == False ) & 
                   (df['pre_experience'].isnull()==False)&(df['missing']==0)]
    work_list = df['work_category'].unique()
    year_list = df['year'].unique()
    outcome = ['discount', 'delay_ratio', 'overrun_ratio', 'days_to_award']

    #iteration
    for o in outcome:

        #make a column list
        reg_col = []
        for j in year_list:
            reg_col.append(j)
        exog_var = ['fpsb_auction']
        exog = exog_var + reg_col 


        #check multicollinearity
        X = df.loc[:,exog]
        vif = calc_vif(X)

        #delete from col list
        for i in range(len(vif)):
            if np.isnan(vif.loc[i, 'VIF']) == True:
                reg_col.remove(vif.loc[i, 'variables'])

        exog = exog_var + reg_col
        exog.remove(2000)

        if o == 'discount':
            fe_reg_discount = mt.reg(df, o, exog, fe_name = 'authority_code', cluster = 'auth_anno',check_colinear = True)
        elif o == 'delay_ratio':
            fe_reg_delay = mt.reg(df, o, exog, fe_name = 'authority_code', cluster = 'auth_anno',check_colinear = True)
        elif o == 'overrun_ratio':
            fe_reg_overrun = mt.reg(df, o, exog, fe_name = 'authority_code', cluster = 'auth_anno',check_colinear = True)
        else :
            fe_reg_award = mt.reg(df, o, exog, fe_name = 'authority_code', cluster = 'auth_anno',check_colinear = True)

    fe_reg = (fe_reg_discount, fe_reg_delay, fe_reg_overrun, fe_reg_award )
    return(fe_reg)


def remove_aba_col2(data):
    #main table
    #col 2
    df = data
    df = df[(df['turin_co_sample']==1)&(df['ctrl_exp_forcedfp_co']==1)&(df['post_experience']>= 5) & (df['pre_experience']>=5) &(df['post_experience'].isnull() == False ) &
                   (df['pre_experience'].isnull()==False)&(df['missing']==0)]
    work_dum = pd.get_dummies(df['work_category']).rename(columns=lambda x: 'work_dum_' + str(x))
    year_dum = pd.get_dummies(df['year']).rename(columns=lambda x: 'year_dum_' + str(x))
    df = pd.concat([df,work_dum],axis = 1)
    df = pd.concat([df,year_dum],axis = 1)

    work_list = df['work_category'].unique()
    year_list = df['year'].unique()

    outcome = ['discount', 'delay_ratio', 'overrun_ratio', 'days_to_award']

    #iteration
    for o in outcome:

        #make a column list
        reg_col = []
        for i in work_list:
            reg_col.append('work_dum_' + i)
        for j in year_list:
            reg_col.append('year_dum_'+ str(j))
        exog_var = ['fpsb_auction','reserve_price','municipality']
        exog = exog_var + reg_col 

        X = df.loc[:,exog]
        vif = calc_vif(X)
        #delete from col list
        for i in range(len(vif)):
            if np.isnan(vif.loc[i, 'VIF']) == True:
                exog.remove(vif.loc[i, 'variables'])
            elif vif.loc[i,'VIF'] > 10:
                for j in exog:
                    if vif.loc[i,'variables'] == j and vif.loc[i,'variables'] != 'fpsb_auction':
                        exog.remove(vif.loc[i,'variables'])

        #exog.remove('year_dum_2000.0')
        #exog.remove('work_dum_OG01')
        exog.remove('municipality')
        exog.remove('work_dum_OS03')
        exog.remove('work_dum_OS04')
        exog.remove('work_dum_OS33')
        exog.remove('work_dum_OG13')
        exog.remove('work_dum_OS09')
        exog.remove('work_dum_OS13')
        exog.remove('work_dum_OS32')
        exog.remove('work_dum_OG09')
        exog.remove('work_dum_OG21')
        exog.remove('work_dum_OG18')
        exog.remove('work_dum_OG24')
        exog.remove('work_dum_OG05')
        exog.remove('year_dum_2008.0')
        exog.remove('year_dum_2009.0')
        exog.remove('year_dum_2007.0')
        exog.remove('year_dum_2010.0')
        #check multicollinearity
        if o == 'discount':

            fe_reg_discount = mt.reg(df, o, exog, fe_name = 'authority_code', cluster = 'auth_anno',addcons= True, check_colinear = True)

        elif o == 'delay_ratio':
            fe_reg_delay = mt.reg(df, o, exog, fe_name = 'authority_code', cluster = 'auth_anno',addcons = True, check_colinear =  True)

        elif o == 'overrun_ratio':
            fe_reg_overrun = mt.reg(df, o, exog, fe_name = 'authority_code', cluster = 'auth_anno',addcons = True, check_colinear = True)

        else :
            fe_reg_award = mt.reg(df, o, exog, fe_name = 'authority_code', cluster = 'auth_anno',addcons = True, check_colinear = True)


    fe_reg = (fe_reg_discount, fe_reg_delay, fe_reg_overrun, fe_reg_award )
    return(fe_reg)

