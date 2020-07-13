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

#df_table = pd.DataFrame({ 'Panel':[], 'value_title':[],'Control(1)':[],'Control(2)':[],'Control(3)':[],'Control(4)':[],'Control(5)':[],'Control(6)':[]})


def reg_tab(parameter, data, column):
    table_string = econ.outreg(parameter, ['fpsb_auction'], ['First Price Auction'], digits = 2)
    table_string += econ.table_statrow("R$^2$", [x.r2 for x in parameter], digits =3)
    table_string += econ.table_statrow("Number of Observation", [x.N for x in parameter])
    table_list = table_string.split('&')
    table_list = [i.split('\\\\ \n',1)[0] for i in table_list]
    table_list.remove(table_list[0])
    
    df_table = data
    value_title = ['First Auction Price','Standard Error','R$^2$','Observations']
    Panel =['A','B','C','D']

    for i in range(len(table_list)):
        df_table.loc[i, 'Control'+'('+str(column)+')'] = table_list[i]
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

    df_table = df_table.sort_values(by='Panel',ascending = True).set_index(['Panel','value_title'])

    return(df_table)

def table4(parameter1, parameter2):
    df_table = pd.DataFrame({ 'value_title':[],'W.Discount(1)':[],'W.Discount(2)':[],'Extra Cost(3)':[],'Extra Cost(4)':[],'Extra Time(5)':[],'Extra Time(6)':[],'Days Award(7)':[],'Days Award(8)':[]})
    
    table_string = econ.outreg(parameter1, ['fpsb_auction'], ['First Price Auction'], digits = 3)
    table_string += econ.table_statrow("R$^2$", [x.r2 for x in parameter1], digits =3)
    table_string += econ.table_statrow("Number of Observation", [x.N for x in parameter1])
    table_list = table_string.split('&')
    table_list = [i.split('\\\\ \n',1)[0] for i in table_list]
    table_list.remove(table_list[0])
    
    col_title_odd = ['W.Discount(1)', 'Extra Cost(3)', 'Extra Time(5)', 'Days Award(7)']
    col_title_even = ['W.Discount(2)','Extra Cost(4)', 'Extra Time(6)', 'Days Award(8)'] 
    value_title = ['First Auction Price','Standard Error','R$^2$','Observations']
    
    for i in range(len(table_list)):
        if i<4:
            df_table.loc[0,'value_title'] = value_title[0]
            df_table.loc[0,col_title_odd[i]] = table_list[i]
        elif i>=4 and i<8:
            df_table.loc[4,'value_title'] = value_title[1]
            df_table.loc[4, col_title_odd[i-4]] = table_list[i]
        elif i>=8 and i<12:
            df_table.loc[8,'value_title'] = value_title[2]
            df_table.loc[8,col_title_odd[i-8]] = table_list[i]
        else:
            df_table.loc[12,'value_title'] = value_title[3]
            df_table.loc[12,col_title_odd[i-12]] = table_list[i]
            
    table_string2 = econ.outreg(parameter2, ['fpsb_auction'], ['First Price Auction'], digits = 3)
    table_string2 += econ.table_statrow("R$^2$", [x.r2 for x in parameter2], digits =3)
    table_string2 += econ.table_statrow("Number of Observation", [x.N for x in parameter2])
    table_list2 = table_string2.split('&')
    table_list2 = [i.split('\\\\ \n',1)[0] for i in table_list2]
    table_list2.remove(table_list2[0])
    
    for i in range(len(table_list2)):
        if i<4:
            df_table.loc[0,'value_title'] = value_title[0]
            df_table.loc[0,col_title_even[i]] = table_list2[i]
        elif i>=4 and i<8:
            df_table.loc[4,'value_title'] = value_title[1]
            df_table.loc[4, col_title_even[i-4]] = table_list2[i]
        elif i>=8 and i<12:
            df_table.loc[8,'value_title'] = value_title[2]
            df_table.loc[8,col_title_even[i-8]] = table_list2[i]
        else:
            df_table.loc[12,'value_title'] = value_title[3]
            df_table.loc[12,col_title_even[i-12]] = table_list2[i]
            
    return(df_table)