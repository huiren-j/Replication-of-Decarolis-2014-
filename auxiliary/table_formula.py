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

def table1_presort(data):
    df_table = pd.DataFrame(index =['Winning discount','Extra cost','Extra time', 'Days to award', 'Reserve price', 'Number of bidders', 'Population', 'Experience', 'Fiscal efficiency'], columns= [['1. Municipality of Turin','1. Municipality of Turin','1. Municipality of Turin','2. County of Turin','2. County of Turin','2. County of Turin', '3. Other PAs','3. Other PAs','3. Other PAs'],['Mean','SD','N','Mean','SD','N','Mean','SD','N']])
    table_values= ['Winning discount','Extra cost','Extra time', 'Days to award', 'Reserve price', 'Number of bidders', 'Population', 'Experience', 'Fiscal efficiency']
    region= ['1. Municipality of Turin','2. County of Turin', '3. Other PAs']
    
    columns = ['discount', 'overrun_ratio', 'delay_ratio', 'days_to_award', 'reserve_price',  'n_bidders', 'population', 'experience','fiscal_efficiency']
    values = ['mean','std','count']
    index = [1.0, 2.0, 3.0]
    
    for i in range(len(columns)):
        for j in range(len(index)):
            #column
            df_table.loc[table_values[i], (region[j],'Mean')] =data.loc[index[j], (columns[i], 'mean')]
            df_table.loc[table_values[i], (region[j],'SD')] =data.loc[index[j], (columns[i], 'std')]
            df_table.loc[table_values[i], (region[j],'N')] =data.loc[index[j], (columns[i], 'count')]

    return(df_table)

def table1_postsort(data):
    df_table = pd.DataFrame(index =['Winning discount','Extra cost','Extra time', 'Days to award', 'Reserve price', 'Number of bidders', 'Population', 'Experience', 'Fiscal efficiency'], columns= [['1. Municipality of Turin','1. Municipality of Turin','1. Municipality of Turin','2. County of Turin','2. County of Turin','2. County of Turin', '3. Other PAs','3. Other PAs','3. Other PAs'],['Mean','SD','N','Mean','SD','N','Mean','SD','N']])
    table_values= ['Winning discount','Extra cost','Extra time', 'Days to award', 'Reserve price', 'Number of bidders', 'Population', 'Experience', 'Fiscal efficiency']
    region= ['1. Municipality of Turin','2. County of Turin', '3. Other PAs']
    
    columns = ['discount', 'overrun_ratio', 'delay_ratio', 'days_to_award', 'reserve_price',  'n_bidders', 'population', 'experience','fiscal_efficiency']
    values = ['mean','std','count']
    index = [1.0, 2.0, 3.0]
    
    for i in range(len(columns)):
        for j in range(len(index)):
            #column
            df_table.loc[table_values[i], (region[j],'Mean')] =data.loc[index[j], (columns[i], 'mean')]
            df_table.loc[table_values[i], (region[j],'SD')] =data.loc[index[j], (columns[i], 'std')]
            df_table.loc[table_values[i], (region[j],'N')] =data.loc[index[j], (columns[i], 'count')]

    return(df_table)

def main_table(parameter):    
    df_table = pd.DataFrame({ 'Panel':[], 'value_title':[],'Control(1)':[],'Control(2)':[],'Control(3)':[],'Control(4)':[],'Control(5)':[],'Control(6)':[]})
    value_title = ['First Price Auction','Standard Error','R$^2$','Observations']
    Panel =['A','B','C','D']
    column_list = ['Control(1)','Control(2)','Control(3)','Control(4)','Control(5)','Control(6)']
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
            if i<4:
                df_table.loc[i,'value_title'] = value_title[0]
            elif i>=4 and i<8:
                df_table.loc[i,'value_title'] = value_title[1]
            elif i>=8 and i<12:
                df_table.loc[i,'value_title'] = value_title[2]
            else:
                df_table.loc[i,'value_title'] = value_title[3]

        for i in range(0,4):
            if i ==0:
                df_table.loc[i,'Panel'] =Panel[i]
                df_table.loc[i+4,'Panel'] =Panel[i]
                df_table.loc[i+8,'Panel'] =Panel[i]
                df_table.loc[i+12,'Panel'] =Panel[i]
            elif i == 1:
                df_table.loc[i,'Panel'] =Panel[i+1]
                df_table.loc[i+4,'Panel'] =Panel[i+1]
                df_table.loc[i+8,'Panel'] =Panel[i+1]
                df_table.loc[i+12,'Panel'] =Panel[i+1]
            elif i == 2:
                df_table.loc[i,'Panel'] =Panel[i-1]
                df_table.loc[i+4,'Panel'] =Panel[i-1]
                df_table.loc[i+8,'Panel'] =Panel[i-1]
                df_table.loc[i+12,'Panel'] =Panel[i-1]
            elif i == 3:
                df_table.loc[i,'Panel'] =Panel[i]
                df_table.loc[i+4,'Panel'] =Panel[i]
                df_table.loc[i+8,'Panel'] =Panel[i]
                df_table.loc[i+12,'Panel'] =Panel[i]
                
        col = col+1

    df_table = df_table.sort_values(by='Panel',ascending = True).set_index(['Panel','value_title'])

    return(df_table)


def table4(parameter1, parameter2):
    df_table = pd.DataFrame({'Panel':[], 'value_title':[],'W.Discount(1)':[],'W.Discount(2)':[],'Extra Cost(3)':[],'Extra Cost(4)':[],'Extra Time(5)':[],'Extra Time(6)':[],'Days Award(7)':[],'Days Award(8)':
                             []})
               
    col_title_odd = ['W.Discount(1)', 'Extra Cost(3)', 'Extra Time(5)', 'Days Award(7)']
    col_title_even = ['W.Discount(2)','Extra Cost(4)', 'Extra Time(6)', 'Days Award(8)'] 
    value_title = ['First Price Auction','Standard Error','R$^2$','Observations']
    Panel = ['A','B']
    
    j = 0  
    k = 0
    for parm in parameter1:
        table_string = econ.outreg(parm, ['fpsb_auction'], ['First Price Auction'], digits = 3)
        table_string += econ.table_statrow("R$^2$", [x.r2 for x in parm], digits =3)
        table_string += econ.table_statrow("Number of Observation", [x.N for x in parm])
        table_list = table_string.split('&')
        table_list = [i.split('\\\\ \n',1)[0] for i in table_list]
        table_list.remove(table_list[0])

        for i in range(len(table_list)):
            if i<4:
                df_table.loc[k,'value_title'] = value_title[0]
                df_table.loc[k,col_title_odd[i]] = table_list[i]
                df_table.loc[k, 'Panel'] = Panel[j]
            elif i>=4 and i<8:
                df_table.loc[k+1,'value_title'] = value_title[1]
                df_table.loc[k+1, col_title_odd[i-4]] = table_list[i]
                df_table.loc[k+1, 'Panel'] = Panel[j]
            elif i>=8 and i<12:
                df_table.loc[k+2,'value_title'] = value_title[2]
                df_table.loc[k+2,col_title_odd[i-8]] = table_list[i]
                df_table.loc[k+2, 'Panel'] = Panel[j]
            else:
                df_table.loc[k+3,'value_title'] = value_title[3]
                df_table.loc[k+3,col_title_odd[i-12]] = table_list[i]
                df_table.loc[k+3, 'Panel'] = Panel[j]
        j = j+1
        k = k+4

    
    j = 0
    k = 0
    for parm in parameter2:
        table_string2 = econ.outreg(parm, ['fpsb_auction'], ['First Price Auction'], digits = 3)
        table_string2 += econ.table_statrow("R$^2$", [x.r2 for x in parm], digits =3)
        table_string2 += econ.table_statrow("Number of Observation", [x.N for x in parm])
        table_list2 = table_string2.split('&')
        table_list2 = [i.split('\\\\ \n',1)[0] for i in table_list2]
        table_list2.remove(table_list2[0])

        for i in range(len(table_list2)):
            if i<4:
                df_table.loc[k,'value_title'] = value_title[0]
                df_table.loc[k,col_title_even[i]] = table_list2[i]
                df_table.loc[k, 'Panel'] = Panel[j]
            elif i>=4 and i<8:
                df_table.loc[k+1,'value_title'] = value_title[1]
                df_table.loc[k+1, col_title_even[i-4]] = table_list2[i]
                df_table.loc[k+1, 'Panel'] = Panel[j]
            elif i>=8 and i<12:
                df_table.loc[k+2,'value_title'] = value_title[2]
                df_table.loc[k+2,col_title_even[i-8]] = table_list2[i]
                df_table.loc[k+2, 'Panel'] = Panel[j]
            else:
                df_table.loc[k+3,'value_title'] = value_title[3]
                df_table.loc[k+3,col_title_even[i-12]] = table_list2[i]
                df_table.loc[k+3, 'Panel'] = Panel[j]
        j = j+1
        k = k+4
    
    df_table = df_table.set_index(['Panel'])
    return(df_table)

def table5_A(parm1, parm2):
    table5_PanelA = pd.DataFrame(index =['PA-Year','PA'], columns=['W.Bid(1)','W.Bid(2)','Extra cost(3)','Extra cost(4)','Extra time(5)','Extra time(6)','Days awrd(7)','Days awrd(8)'])
    for i in range(len(parm1)):
        k=0
        l=1
        for j in range(0,4):
            table5_PanelA.iloc[i,k] = '(' + str(int(parm1.iloc[i,k])) + ';' + str(int(parm1.iloc[i,k+1])) +')'
            table5_PanelA.iloc[i,l] = '(' + str(int(parm2.iloc[i,k])) + ';' + str(int(parm2.iloc[i,k+1])) +')'
            k = k+2
            l= l+2
    return(table5_PanelA)

def table5_B(parm1, parm2):
    table5_PanelB = pd.DataFrame(index =['PA-Year','PA'], columns=['W.Bid(1)','W.Bid(2)','Extra cost(3)','Extra cost(4)','Extra time(5)','Extra time(6)','Days awrd(7)','Days awrd(8)'])
    for i in range(len(parm1)):
        k=0
        l=1
        for j in range(0,4):
            table5_PanelB.iloc[i,k] = '(' + str(int(parm1.iloc[i,k])) + ';' + str(int(parm1.iloc[i,k+1])) +')'
            table5_PanelB.iloc[i,l] = '(' + str(int(parm2.iloc[i,k])) + ';' + str(int(parm2.iloc[i,k+1])) +')'
            k = k+2
            l= l+2
    return(table5_PanelB)

                             
def table6(parameter):
    
    df_table = pd.DataFrame({'Panel':[],'value_title':[],'Base':[], 'Full':[],'Missing':[],'Simple':[],'Complex':[],'2001-2005':[],'2002-2004':[]})
    column_list = ['Base', 'Full','Missing','Simple','Complex','2001-2005','2002-2004']
    col = 0
    value_title = ['First Price Auction','Standard Error','R$^2$','Observations']
    Panel =['A','B']
    
    for parm in parameter:
        table_string = econ.outreg(parm, ['fpsb_auction'], ['First Price Auction'], digits = 2)
        table_string += econ.table_statrow("R$^2$", [x.r2 for x in parm], digits =3)
        table_string += econ.table_statrow("Number of Observation", [x.N for x in parm])
        table_list = table_string.split('&')
        table_list = [i.split('\\\\ \n',1)[0] for i in table_list]
        table_list.remove(table_list[0])

        for i in range(len(table_list)):
            df_table.loc[i , column_list[col]] = table_list[i]
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
            elif i == 1:
                df_table.loc[i,'Panel'] =Panel[i]
                df_table.loc[i+2,'Panel'] =Panel[i]
                df_table.loc[i+4,'Panel'] =Panel[i]
                df_table.loc[i+6,'Panel'] =Panel[i]
        col = col+1


    df_table = df_table.sort_values(by='Panel',ascending = True).set_index(['Panel','value_title'])
    return(df_table)


def table7(dataA, dataB):
    #panelA
    df = dataA
    df_table = pd.DataFrame(index =['PanelA: Voluntary swtiching to FPAs', 'Winning discount', 'Days to award', 'Reserve price', 'Number of bidders', 'Conract Duration', 'Miles PA from Turin', 'Population', 'Experience'], columns= [['AB Auctions','AB Auctions','AB Auctions','AB Auctions','FP Auctions','FP Auctions','FP Auctions','FP Auctions'],['Mean','SD','p50','N','Mean','SD','p50','N']])
    table_values= ['Winning discount', 'Days to award', 'Reserve price', 'Number of bidders', 'Conract Duration', 'Miles PA from Turin', 'Population', 'Experience']
    auction_type= ['AB Auctions','FP Auctions']
    
    columns = [ 'discount','days_to_award','reserve_price', 'n_bidders', 'contract_duration',  'miles_pa_torino', 'population', 'experience' ]
    values = ['mean','std','50%','count']
    index = [0.0, 1.0]
    
    df_table.loc['PanelA: Voluntary swtiching to FPAs', :] = '-'
    
    for i in range(len(columns)):
        for j in index:
            #column
            df_table.loc[table_values[i], (auction_type[int(j)],'Mean')] =df.loc[j, (columns[i], 'mean')]
            df_table.loc[table_values[i], (auction_type[int(j)],'SD')] =df.loc[j, (columns[i], 'std')]
            df_table.loc[table_values[i], (auction_type[int(j)],'p50')] = df.loc[j,(columns[i],'50%')]
            df_table.loc[table_values[i], (auction_type[int(j)],'N')] =df.loc[j, (columns[i], 'count')]
    
    df_tableA = df_table
    
    #panelB
    df = dataB
    df_table = pd.DataFrame(index =['PanelB: Forced to switch to FPAs','Winning discount', 'Days to award', 'Reserve price', 'Number of bidders', 'Conract Duration', 'Miles PA from Turin', 'Population', 'Experience'], columns= [['AB Auctions','AB Auctions','AB Auctions','AB Auctions','FB Auctions','FB Auctions','FB Auctions','FB Auctions'],['Mean','SD','p50','N','Mean','SD','p50','N']])
    table_values= ['Winning discount', 'Days to award', 'Reserve price', 'Number of bidders', 'Conract Duration', 'Miles PA from Turin', 'Population', 'Experience']
    auction_type= ['AB Auctions','FB Auctions']
    
    columns = [ 'discount','days_to_award','reserve_price', 'n_bidders', 'contract_duration',  'miles_pa_torino', 'population', 'experience' ]
    values = ['mean','std','50%','count']
    index = [0.0, 1.0]
    
    df_table.loc['PanelB: Forced to switch to FPAs', :] = '-'
    
    for i in range(len(columns)):
        for j in index:
            #column
            df_table.loc[table_values[i], (auction_type[int(j)],'Mean')] =df.loc[j, (columns[i], 'mean')]
            df_table.loc[table_values[i], (auction_type[int(j)],'SD')] =df.loc[j, (columns[i], 'std')]
            df_table.loc[table_values[i], (auction_type[int(j)],'p50')] = df.loc[j,(columns[i],'50%')]
            df_table.loc[table_values[i], (auction_type[int(j)],'N')] =df.loc[j, (columns[i], 'count')]
    
    df_tableB = df_table
    df_table= pd.concat([df_tableA, df_tableB])
    return(df_table)
