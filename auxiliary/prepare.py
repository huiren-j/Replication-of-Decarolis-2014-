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


def prepare_data(data):
    idx = data[(data['authority_code']==3091058)].index
    df_desc = data.drop(idx)
    #data.shape obs = 16127 = original data length
    idx2 = df_desc[(df_desc['authority_code']!=3090272) & (df_desc['authority_code']!=3070001) & (df_desc['post_ref_adpt'] == 1)].index
    #print(idx2) 1130 obs to be deleted
    df_desc =df_desc.drop(idx2)
    #print(data.shape)
    #(14997,31) checked

    idx = df_desc[(df_desc['reserve_price']>5000000)|(df_desc['reserve_price']<300000)].index
    #print(idx) 6892 obs to be deleted; checked
    df_desc = df_desc.drop(idx)
    #print(df_desc.shape)
    #(8105, 31) checked

    #5101 obs to be deleted
    df_desc = df_desc[(df_desc['ctrl_pop_turin_co_sample']==1) | (df_desc['ctrl_pop_turin_pr_sample']==1) | (df_desc['ctrl_exp_turin_co_sample']==1) | (df_desc['ctrl_exp_turin_pr_sample']==1)]
    #print(df_desc.shape)
    #(3004,31) checked
    
    return(df_desc)

def presort_describe(data):
    df_pre = data
    df_pre['presort'] = np.nan
    df_pre.loc[(df_pre['sample']==0)&(df_pre['pre_experience']>=5),'presort'] = 3
    df_pre.loc[(df_pre['authority_code']==3090272)&(df_pre['presort'].isnull()==False),'presort'] = 1
    df_pre.loc[(df_pre['authority_code']==3070001)& (df_pre['presort'].isnull() ==False),'presort'] = 2

    df_pre =df_pre.filter(items=['presort', 'fiscal_efficiency', 'reserve_price', 'experience', 'population', 'discount', 'n_bidders', 'overrun_ratio', 'delay_ratio', 'days_to_award'])
    idx = df_pre[df_pre['presort'].isnull() == True].index
    df_pre = df_pre.drop(idx)
    df_pre['reserve_price']= df_pre['reserve_price']/1000
    df_pre['population']=df_pre['population']/1000
    
    #descriptive
    pre_describe = df_pre.groupby('presort')[['discount', 'overrun_ratio', 'delay_ratio', 'days_to_award', 'reserve_price',  'n_bidders', 'population', 'experience',
                                              'fiscal_efficiency']].describe()
    
    return(pre_describe)

def postsort_describe(data):
    df_post = data
    df_post['postsort'] = np.nan
    df_post.loc[(df_post['sample']==1)&(df_post['post_experience']>=5),'postsort'] = 3
    df_post.loc[(df_post['authority_code']==3090272)&(df_post['postsort'].isnull()==False),'postsort'] = 1
    df_post.loc[(df_post['authority_code']==3070001)& (df_post['postsort'].isnull() ==False),'postsort'] = 2

    df_post =df_post.filter(items=['postsort', 'fiscal_efficiency', 'reserve_price', 'experience', 'population', 'discount', 'n_bidders', 'overrun_ratio', 'delay_ratio', 'days_to_award'])
    df_post['reserve_price']= df_post['reserve_price']/1000 #rp = three digits or four
    df_post['population']=df_post['population']/1000

    idx = df_post[df_post['postsort'].isnull() == True].index
    df_post = df_post.drop(idx)
    df_post.head() # (1223,10)

    #descriptive
    post_describe = df_post.groupby('postsort')[['discount', 'overrun_ratio', 'delay_ratio', 'days_to_award', 'reserve_price',  'n_bidders', 'population', 'experience',
                                                 'fiscal_efficiency']].describe()
    
    return(post_describe)


def basic_setting(data):
    #construct work category dummy
    df = data
    df['OG03_dummy'] = 0
    df.loc[(df['work_category']=='OG03')&(df['work_category']!=''),'OG03_dummy'] = 1
    
    df['OG01_dummy'] = 0
    df.loc[(df['work_category']=='OG01')&(df['work_category']!=''),'OG01_dummy'] = 1
    
    df['OG_rest_dummy'] = 0
    df.loc[(df['OG01_dummy']!=1)&(df['OG03_dummy']!=1)&(df['work_category']!=''),'OG_rest_dummy'] = 1
    
    df['OG_dummy'] = 0
    df.loc[df['work_category'].str[0:2] == 'OG','OG_dummy'] = 1
    
    df['OS_dummy'] = 0
    df.loc[df['work_category'].str[0:2] == 'OS','OS_dummy'] = 1
    
    #treated vs controls
    df['trend'] = df['year'] - 1999

    df['trend_treat'] = df['trend']
    df.loc[(df['authority_code']!=3090272)&(df['authority_code']!=3070001),'trend_treat'] = 0

    df['trend_control'] = df['trend']
    df.loc[(df['authority_code']==3090272)|(df['authority_code']==3070001),'trend_control'] = 0

    #PA specifics
    df = df.sort_values(by='authority_code',ascending=True)

    auth_list = df['authority_code'].values.tolist()
    auth_list = list(set(auth_list))
    df['id_auth'] = 0
    for i in range(len(df)):
        for j in range(len(auth_list)):
            if df.loc[i,'authority_code'] == auth_list[j]:
                df.loc[i,'id_auth'] = j+1

    work_dum = pd.get_dummies(df['work_category'])
    year_dum = pd.get_dummies(df['year'])
    work_list = list(work_dum.columns)
    year_list = list(year_dum.columns)

    df_dum = pd.concat([year_dum, work_dum],axis = 1)
    df = pd.concat([df, df_dum],axis = 1)
    
    return(df)

def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)
