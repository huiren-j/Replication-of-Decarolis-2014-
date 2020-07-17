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

def multi_collinearity(data):
    df = data
    df0 = df[(df['ctrl_exp_turin_co_sample']==1)|(df['ctrl_pop_turin_co_sample']==1)|(df['ctrl_exp_turin_pr_sample']==1)|(df['ctrl_exp_turin_pr_sample']==1)]
    plt.figure(figsize=(13, 10), dpi=70)
    plt.subplot(231)
    plt.scatter(df0['municipality'], df0['fpsb_auction'])
    plt.xlabel('Municipality')
    plt.ylabel('First Price Auction')
    
    plt.subplot(232)
    plt.scatter(df0['municipality'], df0['fiscal_efficiency'])
    plt.xlabel('Municipality')
    plt.ylabel('Fiscal Efficiency')
    
    plt.subplot(233)
    plt.scatter(df0['municipality'], df0['reserve_price'])
    plt.xlabel('Municipality')
    plt.ylabel('Reserve Price')
    
    plt.subplot(234)
    plt.scatter(df0['fpsb_auction'], df0['reserve_price'])
    plt.ylabel('Reserve Price')
    plt.xlabel('First Price Auction')
    
    plt.subplot(235)
    plt.scatter(df0['fpsb_auction'],df0['fiscal_efficiency'])
    plt.ylabel('Fiscal Efficiency')
    plt.xlabel('First Price Auction')
    
    plt.subplot(236)
    plt.scatter(df0['fiscal_efficiency'],df0['reserve_price'])
    plt.ylabel('Reserve Price')
    plt.xlabel('Fiscal Efficiency')
    
    return(plt.show())

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


def population_check(data1, data2):
    df = data1
    df_ext = data2
    pop_check = []
    pop_check.append(df_ext.loc[df_ext['ctrl_exp_adopters_vlunt']==1,'population'].mean())
    pop_check.append(df.loc[df['ctrl_exp_turin_co_sample']==1, 'population'].mean())
    pop_check.append(df.loc[df['ctrl_exp_turin_pr_sample']==1, 'population'].mean())

    plt.bar(['Voluntary Switcher','Municipality of Turin','County of Turin'],pop_check,color='c')
    plt.xlabel('Standard PAs to Define Control Group')
    plt.ylabel('Population')
    plt.title('Comparison of Population between Control Groups')
    
    return(plt.show())
