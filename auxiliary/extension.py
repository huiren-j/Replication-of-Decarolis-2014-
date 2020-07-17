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

'''extension with descriptive statistics of IE'''

def plot_discount_IE(data, table):
    data2 = data
    tab7 = table
    plt.figure()
    plt.hist(data2.loc[(data2['adopters_vlunt']==1), 'discount'], bins = 30, color='orange', alpha=0.7 )
    plt.axvline(x= tab7.iloc[0,tab7.columns.get_loc(('AB Auctions','Mean'))], color='r')
    plt.axvline(x= tab7.iloc[0,tab7.columns.get_loc(('FB Auctions','Mean'))], color='b')
    plt.xlabel('Winning discount for voluntary switcher group')
    plt.ylabel('Freq.')
    plt.title('Distribution of winning discount of voluntary switcher group')

    plt.figure()
    plt.hist(data2.loc[(data2['forcedfp_strict']==1), 'discount'], bins = 30, color='green', alpha=0.7 )
    plt.axvline(x= tab7.iloc[8,tab7.columns.get_loc(('AB Auctions','Mean'))], color='r')
    plt.axvline(x= tab7.iloc[8,tab7.columns.get_loc(('FB Auctions','Mean'))], color='b')
    plt.xlabel('Winning discount for forced switcher group')
    plt.ylabel('Freq.')
    plt.title('Distribution of winning discount of forced switcher group')

    return(plt.show)

def plot_screening_IE(data, table):
    data2 = data
    tab7 = table
    plt.figure()
    plt.xlim(-5, 175)
    plt.hist(data2.loc[(data2['adopters_vlunt']==1), 'days_to_award'], bins = 30, color='orange', alpha=0.7 )
    plt.axvline(x= tab7.iloc[0,tab7.columns.get_loc(('AB Auctions','Mean'))], color='r')
    plt.axvline(x= tab7.iloc[0,tab7.columns.get_loc(('FB Auctions','Mean'))], color='b')
    plt.xlabel('Screening cost for voluntary switcher group')
    plt.ylabel('Freq.')
    plt.title('Distribution of days to award contract of voluntary switcher group')

    plt.figure()
    plt.xlim(-5, 175)
    plt.ylim(0, 70)
    plt.hist(data2.loc[(data2['forcedfp_strict']==1), 'days_to_award'], bins = 30, color='green', alpha=0.7 )
    plt.axvline(x= tab7.iloc[8,tab7.columns.get_loc(('AB Auctions','Mean'))], color='r')
    plt.axvline(x= tab7.iloc[8,tab7.columns.get_loc(('FB Auctions','Mean'))], color='b')
    plt.xlabel('Screening cost for forced switcher group')
    plt.ylabel('Freq.')
    plt.title('Distribution of days to award contract of forced switcher group')
    
    return(plt.show)

'''Regression with IE'''

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

def table_ext(parameter):    
    df_table = pd.DataFrame({'Panel':[],'value_title':[],'Control(1)':[],'Control(2)':[]})
    value_title = ['First Price Auction','Standard Error','R$^2$','Observations']
    column_list = ['Control(1)','Control(2)']
    Panel =['A. Winning discount','B. Days to award the contract']
    col = 0
    
    for parm in parameter:    
        table_string = econ.outreg(parm, ['fpsb_auction'], ['First Price Auction'], digits = 2)
        table_string += econ.table_statrow("R$^2$", [x.r2 for x in parm], digits =3)
        table_string += econ.table_statrow("Number of Observation", [x.N for x in parm])
        table_list = table_string.split('&')
        table_list = [i.split('\\\\ \n',1)[0] for i in table_list]
        table_list.remove(table_list[0])

        for i in range(len(table_list)):
            df_table.loc[i, column_list[col]] = table_list[i]
            if i<2:
                df_table.loc[i,'value_title'] = value_title[0]
            elif i>=2 and i<4:
                df_table.loc[i,'value_title'] = value_title[1]
            elif i>=4 and i<6:
                df_table.loc[i,'value_title'] = value_title[2]
            else:
                df_table.loc[i,'value_title'] = value_title[3]
        
        for i in range(0,2):
            if i ==0:
                df_table.loc[i,'Panel'] =Panel[i]
                df_table.loc[i+2,'Panel'] =Panel[i]
                df_table.loc[i+4,'Panel'] =Panel[i]
                df_table.loc[i+6,'Panel'] =Panel[i]
            else:
                df_table.loc[i,'Panel'] =Panel[i]
                df_table.loc[i+2,'Panel'] =Panel[i]
                df_table.loc[i+4,'Panel'] =Panel[i]
                df_table.loc[i+6,'Panel'] =Panel[i]
                
        col = col+1

    df_table = df_table.sort_values(by='Panel',ascending = True).set_index(['Panel','value_title'])

    return(df_table)


def plot_comparions_to_baseline(table):

    tab8 = table.reset_index()

    dis_1= float(tab8.iloc[0,2][:5])
    dis_2= float(tab8.iloc[0,3][:5])
    day_1= float(tab8.iloc[4,2][:6])
    day_2= float(tab8.iloc[4,3][:6])

    dis_tab8 = [dis_1, dis_2]
    dis_tab2 = [13.10, 11.99]
    dis_tab3 = [8.65, 8.69]
    day_tab8 = [day_1, day_2]
    day_tab2 = [28.70, 26.23]
    day_tab3 = [35.02, 37.49]

    #plot
    plt.figure()
    plt.ylim(5,15)
    plt.plot(dis_tab8,'o',color = 'r')
    plt.plot(dis_tab2,'o',color = 'b')
    plt.plot(dis_tab3,'o',color = 'orange')
    plt.xlabel('Firs Price Auction')
    plt.ylabel('The Effect on Winning Discount')
    plt.title('Comparison to the estimated in Table2 and Table3: Winning Discount')

    plt.figure()
    plt.ylim(20,40)
    plt.plot(day_tab8,'o',color = 'r')
    plt.plot(day_tab2,'o',color = 'b')
    plt.plot(day_tab3,'o',color = 'orange')
    plt.xlabel('Firs Price Auction')
    plt.ylabel('The Effect on Days to Award')
    plt.title('Comparison to the estimated in Table2 and Table3: Days to Award')
    return(plt.show())

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

