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

def table7_PanelA(data):
    #descriptive
    df = data
    df = df[df['forcedfp_strict']==0]
    df = df.reset_index()
    df['lmiles_pa_to'] = np.log(df['miles_pa_torino']+1)
    del df['experience']

    df = df.sort_values(by = 'authority_code', ascending = True)
    exp = df['authority_code'].value_counts()
    exp = pd.DataFrame({'auth':exp.index, 'value':exp.values})

    for i in range(len(df)):
        for j in range(len(exp)):
            if df.loc[i,'authority_code'] == exp.loc[j, 'auth']:
                df.loc[i, 'experience'] = exp.loc[j, 'value']

    df['reserve_price'] = df['reserve_price']/1000
    df['population'] = df['population']/1000
    df= df[['fpsb_auction', 'forcedfp_strict', 'reserve_price', 'contract_duration', 'experience', 'population', 'discount', 'n_bidders', 'miles_pa_torino', 'miles_firm_work', 'days_to_award']]
    #idx = df[df['forcedfp_strict']!=1].index
    #df = df.drop(idx)
    
    describe = df.groupby('fpsb_auction')[['reserve_price', 'days_to_award', 'contract_duration', 'miles_pa_torino', 'experience', 'population', 'discount', 'n_bidders']].describe().round(2)
    return(describe)

def table7_PanelB(data):
    #descriptive
    df = data
    df = df[df['forcedfp_strict']==1]
    df = df.reset_index()
    df['lmiles_pa_to'] = np.log(df['miles_pa_torino']+1)
    del df['experience']

    df = df.sort_values(by = 'authority_code', ascending = True)
    exp = df['authority_code'].value_counts()
    exp = pd.DataFrame({'auth':exp.index, 'value':exp.values})

    for i in range(len(df)):
        for j in range(len(exp)):
            if df.loc[i,'authority_code'] == exp.loc[j, 'auth']:
                df.loc[i, 'experience'] = exp.loc[j, 'value']

    df['reserve_price'] = df['reserve_price']/1000
    df['population'] = df['population']/1000
    df= df[['fpsb_auction', 'forcedfp_strict', 'reserve_price', 'contract_duration', 'experience', 'population', 'discount', 'n_bidders', 'miles_pa_torino', 'miles_firm_work', 'days_to_award']]
    #idx = df[df['forcedfp_strict']!=1].index
    #df = df.drop(idx)
    
    describe = df.groupby('fpsb_auction')[['reserve_price', 'days_to_award', 'contract_duration', 'miles_pa_torino', 'experience', 'population', 'discount', 'n_bidders']].describe().round(2)
    return(describe)
