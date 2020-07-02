def prepare_data(data):
    idx = data[(data['authority_code']==3091058)].index
    df_desc = data.drop(idx)
    #data.shape obs = 16127 = original data length
    idx2 = df_desc[(df_desc['authority_code']!=3090272) & (df_desc['authority_code']!=3070001) & (df_desc['post_ref_adpt'] == 1)].index
    #print(idx2) 1130 obs to be deleted
    df_desc =df_desc.drop(idx2)
    print(data.shape)
    #(14997,31) checked

    idx = df_desc[(df_desc['reserve_price']>5000000)|(df_desc['reserve_price']<300000)].index
    #print(idx) 6892 obs to be deleted; checked
    df_desc = df_desc.drop(idx)
    print(df_desc.shape)
    #(8105, 31) checked

    #5101 obs to be deleted
    df_desc = df_desc[(df_desc['ctrl_pop_turin_co_sample']==1) | (df_desc['ctrl_pop_turin_pr_sample']==1) | (df_desc['ctrl_exp_turin_co_sample']==1) | (df_desc['ctrl_exp_turin_pr_sample']==1)]
    print(df_desc.shape)
    #(3004,31) checked
    
    return df_desc

def presort_describe(data):
    df_pre = df_desc
    df_pre['presort'] = np.nan
    df_pre.loc[(df_pre['sample']==0)&(df_pre['pre_experience']>=5),'presort'] = 3
    #idx = df[(df['presort'].isnull()==True)].index
    #print(idx) 1811 obs = NaN checked

    #realnm = df[df['presort']==3].index
    #print(realnm); starting point, 1193 obs were presrot=3  

    df_pre.loc[(df_pre['authority_code']==3090272)&(df_pre['presort'].isnull()==False),'presort'] = 1
    #realnm = df[df['presort']==1].index
    #print(realnm) 
    ##121 obs -> presort ==1 chekced

    df_pre.loc[(df_pre['authority_code']==3070001)& (df_pre['presort'].isnull() ==False),'presort'] = 2
    #realnm = df[df['presort']==2].index
    #print(realnm) 
    #63 obs -> presort ==2 checked
    #print(df.shape) = (3004,32)

    df_pre =df_pre.filter(items=['presort', 'fiscal_efficiency', 'reserve_price', 'experience', 'population', 'discount', 'n_bidders', 'overrun_ratio', 'delay_ratio', 'days_to_award'])
    idx = df_pre[df_pre['presort'].isnull() == True].index
    df_pre = df_pre.drop(idx)
    #print(df.shape) #1181 obs dropped, 1193 left (1193,10) checked

    df_pre['reserve_price']= df_pre['reserve_price']/1000 #rp = three digits or four
    df_pre['population']=df_pre['population']/1000
    #df.head()
    print(df_pre.shape) 
    #(1193,10) checked

    #descriptive
    df_pre.groupby('presort')[['discount', 'overrun_ratio', 'delay_ratio', 'days_to_award', 'reserve_price',  'n_bidders', 'population', 'experience', 'fiscal_efficiency']].describe()
    
    return df_pre