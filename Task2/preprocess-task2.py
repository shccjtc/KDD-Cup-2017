#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues April 14 20:39:10 2020

@author: Song Huancheng
放在/data5002/的目录下，
数据存放路径为 './train/process_data_task2.csv' 以及 './test/process_data_task2.csv'
全局统计数据路径为 'global_statistics/task2'
"""

import pandas as pd
import numpy as np
import copy
import itertools
from datetime import datetime, timedelta, date, time

# ----------------------------- 读取文件 ----------------------------------
def getTime(time):
    left_time = time.replace(minute=(int)(time.minute/20)*20,second=0)
    return str(left_time)
# 提取4-10, 13-19点数据，过滤10.1-10.7数据，统计y
def readData(output_file,is_train):
    df = pd.read_csv(output_file)
    df = pd.read_csv(output_file)
    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour
    df['date'] = df['time'].dt.date
    df['start_time'] = df['time'].map(lambda x: getTime(x))
    first_2hours = [4, 10, 13, 19]  # 早上6小时，下午6小时
    ind1 = (df['hour'] >= first_2hours[0]) & (df['hour'] < first_2hours[1])
    ind2 = (df['hour'] >= first_2hours[2]) & (df['hour'] < first_2hours[3])
    ind3 = df.date.map(str).isin(['2016-10-01', '2016-10-02', '2016-10-03', '2016-10-04', '2016-10-05', '2016-10-06', '2016-10-07'])
    ind = (ind1 | ind2) & (~ind3)
    df = df[ind]

    df['tollgate_id'] = df['tollgate_id'].map(str)
    df['direction'] = df['direction'].map(str)
    df['start_time_20min'] = df['start_time'].apply(lambda x: x.split(' ')[1])
    df_mean = df.groupby(['tollgate_id', 'start_time','direction']).size().reset_index(name='y')
    # print('len origin y:',len(df_mean))

    alldata = list(df.date.map(str).unique())
    alltime = list(df.start_time_20min.map(str).unique())

    all_start_time = []
    for x, y in itertools.product(alldata, alltime):
        all_start_time.append(x + ' ' + str(y))

    all_start_time_tollgate_direction = []
    for x in itertools.product(['1','3'], all_start_time, ['1']):
        all_start_time_tollgate_direction.append(x)
    for x in itertools.product(['1','2','3'], all_start_time, ['0']):
        all_start_time_tollgate_direction.append(x)

    df_y = pd.DataFrame(all_start_time_tollgate_direction, columns=['tollgate_id', 'start_time','direction'])
    df_y = pd.merge(df_y, df_mean, how='left', on=['tollgate_id', 'start_time','direction'])
    df_y = df_y.fillna(method="ffill")
    # print('len df_y',len(df_y))
    df = df.drop(['hour','vehicle_type'],axis=1)


    # 展开one-hot
    all_vmodel = list(df.vehicle_model.map(str).unique())
    all_etc = list(df.has_etc.map(str).unique())
    df = pd.get_dummies(df,columns=['vehicle_model','has_etc'])

    # group统计各vehicle_model, has_etc的数量，
    df = df.groupby(['tollgate_id','direction','start_time']).sum().reset_index()
    
    # 处理缺失值
    df = pd.merge(df_y,df,how='left',on=['tollgate_id','direction','start_time'])
    df = df.fillna(method='ffill')

    # 处理测试集
    if not is_train:
        alldata = ['2016-10-18','2016-10-19','2016-10-20','2016-10-21','2016-10-22','2016-10-23','2016-10-24']

        alltime = ['08:00:00','08:20:00','08:40:00','09:00:00','09:20:00','09:40:00'\
                ,'17:00:00','17:20:00','17:40:00','18:00:00','18:20:00','18:40:00']
                

        all_start_time = []
        for x, y in itertools.product(alldata, alltime):
            all_start_time.append(x + ' ' + str(y))

        all_start_time_tollgate_direction = []
        for x in itertools.product(['1','3'], all_start_time, ['1']):
            all_start_time_tollgate_direction.append(x)
        for x in itertools.product(['1','2','3'], all_start_time, ['0']):
            all_start_time_tollgate_direction.append(x)

        df2 = pd.DataFrame(all_start_time_tollgate_direction, columns=['tollgate_id', 'start_time','direction'])
        df = pd.merge(df, df2, how='outer', on=['tollgate_id', 'start_time','direction'])
        df = df.fillna(0)
    

    df['tollgate_direction'] = df['tollgate_id'] + df['direction']
    df = pd.get_dummies(df,columns=['tollgate_direction'])

    return df, all_vmodel, all_etc





# ----------------------------- rolling prev 2h data ------------------------------
def pre2hBaseData(df2,all_vmodel,all_etc):

    df2['hour_min'] = df2['start_time'].map(lambda x: x.split(' ')[1])

    cond1 = (df2['tollgate_id'] == '1') & (df2['direction'] == '0')
    cond2 = (df2['tollgate_id'] == '1') & (df2['direction'] == '1')
    cond3 = (df2['tollgate_id'] == '2') & (df2['direction'] == '0')
    cond4 = (df2['tollgate_id'] == '3') & (df2['direction'] == '0')
    cond5 = (df2['tollgate_id'] == '3') & (df2['direction'] == '1')
    cond_list = [cond1,cond2,cond3,cond4,cond5]



    ind_list = [('08:00:00','17:00:00'),\
                ('08:20:00','17:20:00'),\
                ('08:40:00','17:40:00'),\
                ('09:00:00','18:00:00'),\
                ('09:20:00','18:20:00'),\
                ('09:40:00','18:40:00')]

    all_vmodel.sort()
    df3 = pd.DataFrame()

    # distance from 1 - 6， 表示从每条数据的前distance条开始读取roll data
    for distance in range(1,7):
        # 按照tollgate_id以及direction分成5个子集
        l_time, r_time = ind_list[distance-1]
        ind = (df2['hour_min']<=l_time) | ((df2['hour_min']>='15:00:00') & (df2['hour_min']<=r_time))
        for cond in cond_list:
            df_cond = df2[cond & ind]
            df_cond = df_cond.sort_values(by=['tollgate_id','direction','start_time'])
            for i in range(1,7):
                for vtype in all_vmodel:
                    df_cond['vehicle_model_'+vtype+'-last_'+str(i)] = df_cond['vehicle_model_'+vtype].shift(i+distance-1,fill_value=0)
                for etc in all_etc:
                    df_cond['has_etc_'+etc+'-last_'+str(i)] = df_cond['has_etc_'+etc].shift(i+distance-1,fill_value=0)
                df_cond['y-last_'+str(i)] = df_cond['y'].shift(i+distance-1,fill_value=0) 
            #保存 distance
            df_cond['distance'] = distance
            #计算mean,sum,median
            for vtype in all_vmodel:
                df_cond['vehicle_model_'+vtype+'_mean'] = df_cond[['vehicle_model_'+vtype+'-last_1','vehicle_model_'+vtype+'-last_2'\
                                                                ,'vehicle_model_'+vtype+'-last_3','vehicle_model_'+vtype+'-last_4'\
                                                                ,'vehicle_model_'+vtype+'-last_5','vehicle_model_'+vtype+'-last_6']].mean(1)
                df_cond['vehicle_model_'+vtype+'_sum'] = df_cond[['vehicle_model_'+vtype+'-last_1','vehicle_model_'+vtype+'-last_2'\
                                                                ,'vehicle_model_'+vtype+'-last_3','vehicle_model_'+vtype+'-last_4'\
                                                                ,'vehicle_model_'+vtype+'-last_5','vehicle_model_'+vtype+'-last_6']].sum(1)
                df_cond['vehicle_model_'+vtype+'_median'] = df_cond[['vehicle_model_'+vtype+'-last_1','vehicle_model_'+vtype+'-last_2'\
                                                                ,'vehicle_model_'+vtype+'-last_3','vehicle_model_'+vtype+'-last_4'\
                                                                ,'vehicle_model_'+vtype+'-last_5','vehicle_model_'+vtype+'-last_6']].median(1)
                df_cond['vehicle_model_'+vtype+'_std'] = df_cond[['vehicle_model_'+vtype+'-last_1','vehicle_model_'+vtype+'-last_2'\
                                                                ,'vehicle_model_'+vtype+'-last_3','vehicle_model_'+vtype+'-last_4'\
                                                                ,'vehicle_model_'+vtype+'-last_5','vehicle_model_'+vtype+'-last_6']].std(1)
                
            for etc in all_etc:
                df_cond['has_etc_'+etc+'_mean'] = df_cond[['has_etc_'+etc+'-last_1','has_etc_'+etc+'-last_2'\
                                                        ,'has_etc_'+etc+'-last_3','has_etc_'+etc+'-last_4'\
                                                        ,'has_etc_'+etc+'-last_5','has_etc_'+etc+'-last_6']].mean(1)
                df_cond['has_etc_'+etc+'_sum'] = df_cond[['has_etc_'+etc+'-last_1','has_etc_'+etc+'-last_2'\
                                                        ,'has_etc_'+etc+'-last_3','has_etc_'+etc+'-last_4'\
                                                        ,'has_etc_'+etc+'-last_5','has_etc_'+etc+'-last_6']].sum(1)
                df_cond['has_etc_'+etc+'_median'] = df_cond[['has_etc_'+etc+'-last_1','has_etc_'+etc+'-last_2'\
                                                        ,'has_etc_'+etc+'-last_3','has_etc_'+etc+'-last_4'\
                                                        ,'has_etc_'+etc+'-last_5','has_etc_'+etc+'-last_6']].median(1)
                df_cond['has_etc_'+etc+'_std'] = df_cond[['has_etc_'+etc+'-last_1','has_etc_'+etc+'-last_2'\
                                                        ,'has_etc_'+etc+'-last_3','has_etc_'+etc+'-last_4'\
                                                        ,'has_etc_'+etc+'-last_5','has_etc_'+etc+'-last_6']].std(1)

            df_cond['y_prev2h_mean'] = df_cond[['y-last_1','y-last_2','y-last_3','y-last_4','y-last_5','y-last_6']].mean(1)
            df_cond['y_prev2h_sum'] = df_cond[['y-last_1','y-last_2','y-last_3','y-last_4','y-last_5','y-last_6']].sum(1)
            df_cond['y_prev2h_median'] = df_cond[['y-last_1','y-last_2','y-last_3','y-last_4','y-last_5','y-last_6']].median(1)
            df_cond['y_prev2h_std'] = df_cond[['y-last_1','y-last_2','y-last_3','y-last_4','y-last_5','y-last_6']].std(1)

            c_ind = (df_cond['hour_min']==l_time) | (df_cond['hour_min']==r_time)
            df3 = df3.append(df_cond[c_ind])
    return df3.drop(['hour_min'],axis=1)

def prev2hData(df2,all_vmodel,all_etc):

    cond1 = (df2['tollgate_id'] == '1') & (df2['direction'] == '0')
    cond2 = (df2['tollgate_id'] == '1') & (df2['direction'] == '1')
    cond3 = (df2['tollgate_id'] == '2') & (df2['direction'] == '0')
    cond4 = (df2['tollgate_id'] == '3') & (df2['direction'] == '0')
    cond5 = (df2['tollgate_id'] == '3') & (df2['direction'] == '1')
    cond_list = [cond1,cond2,cond3,cond4,cond5]
    all_vmodel.sort()
    df3 = pd.DataFrame()

    # distance from 1 - 6， 表示从每条数据的前distance条开始读取roll data
    for distance in range(1,7):
        # 按照tollgate_id以及direction分成5个子集
        for cond in cond_list:
            df_cond = df2[cond]
            df_cond = df_cond.sort_values(by=['tollgate_id','direction','start_time'])
            # 读取前 i*20min的数据
            for i in range(1,7):
                for vtype in all_vmodel:
                    df_cond['vehicle_model_'+vtype+'-last_'+str(i)] = df_cond['vehicle_model_'+vtype].shift(i+distance-1,fill_value=0)
                for etc in all_etc:
                    df_cond['has_etc_'+etc+'-last_'+str(i)] = df_cond['has_etc_'+etc].shift(i+distance-1,fill_value=0)
                df_cond['y-last_'+str(i)] = df_cond['y'].shift(i+distance-1,fill_value=0) 
            #保存 distance
            df_cond['distance'] = distance
            #计算mean,sum,median,std
            for vtype in all_vmodel:
                df_cond['vehicle_model_'+vtype+'_mean'] = df_cond[['vehicle_model_'+vtype+'-last_1','vehicle_model_'+vtype+'-last_2'\
                                                                ,'vehicle_model_'+vtype+'-last_3','vehicle_model_'+vtype+'-last_4'\
                                                                ,'vehicle_model_'+vtype+'-last_5','vehicle_model_'+vtype+'-last_6']].mean(1)
                df_cond['vehicle_model_'+vtype+'_sum'] = df_cond[['vehicle_model_'+vtype+'-last_1','vehicle_model_'+vtype+'-last_2'\
                                                                ,'vehicle_model_'+vtype+'-last_3','vehicle_model_'+vtype+'-last_4'\
                                                                ,'vehicle_model_'+vtype+'-last_5','vehicle_model_'+vtype+'-last_6']].sum(1)
                df_cond['vehicle_model_'+vtype+'_median'] = df_cond[['vehicle_model_'+vtype+'-last_1','vehicle_model_'+vtype+'-last_2'\
                                                                ,'vehicle_model_'+vtype+'-last_3','vehicle_model_'+vtype+'-last_4'\
                                                                ,'vehicle_model_'+vtype+'-last_5','vehicle_model_'+vtype+'-last_6']].median(1)
                df_cond['vehicle_model_'+vtype+'_std'] = df_cond[['vehicle_model_'+vtype+'-last_1','vehicle_model_'+vtype+'-last_2'\
                                                                ,'vehicle_model_'+vtype+'-last_3','vehicle_model_'+vtype+'-last_4'\
                                                                ,'vehicle_model_'+vtype+'-last_5','vehicle_model_'+vtype+'-last_6']].std(1)
                
            for etc in all_etc:
                df_cond['has_etc_'+etc+'_mean'] = df_cond[['has_etc_'+etc+'-last_1','has_etc_'+etc+'-last_2'\
                                                        ,'has_etc_'+etc+'-last_3','has_etc_'+etc+'-last_4'\
                                                        ,'has_etc_'+etc+'-last_5','has_etc_'+etc+'-last_6']].mean(1)
                df_cond['has_etc_'+etc+'_sum'] = df_cond[['has_etc_'+etc+'-last_1','has_etc_'+etc+'-last_2'\
                                                        ,'has_etc_'+etc+'-last_3','has_etc_'+etc+'-last_4'\
                                                        ,'has_etc_'+etc+'-last_5','has_etc_'+etc+'-last_6']].sum(1)
                df_cond['has_etc_'+etc+'_median'] = df_cond[['has_etc_'+etc+'-last_1','has_etc_'+etc+'-last_2'\
                                                        ,'has_etc_'+etc+'-last_3','has_etc_'+etc+'-last_4'\
                                                        ,'has_etc_'+etc+'-last_5','has_etc_'+etc+'-last_6']].median(1)
                df_cond['has_etc_'+etc+'_std'] = df_cond[['has_etc_'+etc+'-last_1','has_etc_'+etc+'-last_2'\
                                                        ,'has_etc_'+etc+'-last_3','has_etc_'+etc+'-last_4'\
                                                        ,'has_etc_'+etc+'-last_5','has_etc_'+etc+'-last_6']].std(1)
            
            df_cond['y_prev2h_mean'] = df_cond[['y-last_1','y-last_2','y-last_3','y-last_4','y-last_5','y-last_6']].mean(1)
            df_cond['y_prev2h_sum'] = df_cond[['y-last_1','y-last_2','y-last_3','y-last_4','y-last_5','y-last_6']].sum(1)
            df_cond['y_prev2h_median'] = df_cond[['y-last_1','y-last_2','y-last_3','y-last_4','y-last_5','y-last_6']].median(1)
            df_cond['y_prev2h_std'] = df_cond[['y-last_1','y-last_2','y-last_3','y-last_4','y-last_5','y-last_6']].std(1)


                # print(distance,i,df_cond['y-last_'+str(i)].isnull().any())
            df3 = df3.append(df_cond)
    
    # 过滤得到8-10，17-19点的数据
    df3['time'] = pd.to_datetime(df3['start_time'])
    df3['hour'] = df3['time'].dt.hour
    # df3['date'] = df3['time'].dt.date

    first_2hours = [8, 10, 17, 19]  # 早上2小时，下午2小时
    ind1 = (df3['hour'] >= first_2hours[0]) & (df3['hour'] < first_2hours[1])
    ind2 = (df3['hour'] >= first_2hours[2]) & (df3['hour'] < first_2hours[3])
    ind = ind1 | ind2
    process_data = df3[ind]
    return process_data.drop(['time','hour'],axis=1)


# ----------------------------------- 统计全局y信息 --------------------------------
def processGlobal(process_data):
    process_data['hour_min'] = process_data['start_time'].map(lambda x: x.split(' ')[1])
    process_data['weekday'] = pd.to_datetime(process_data['start_time']).dt.weekday

    # 第一次运行时需要解注释下面两块内容

    # hcd_mean = process_data.groupby(['hour_min'])['y'].mean().reset_index().reset_index().rename(columns={'y':'y_global20mins_mean','index':'timewindow_id'})
    # hcd_median = process_data.groupby(['hour_min'])['y'].median().reset_index().rename(columns={'y':'y_global20mins_median'})
    # wcd_mean = process_data.groupby(['weekday'])['y'].mean().reset_index().rename(columns={'y':'y_globalweekday_mean'})
    # wcd_median = process_data.groupby(['weekday'])['y'].median().reset_index().rename(columns={'y':'y_globalweekday_median'})

    # hcd_mean.to_csv('./global_statistics/task2/hcd_mean.csv',index=False)
    # hcd_median.to_csv('./global_statistics/task2/hcd_median.csv',index=False)
    # wcd_mean.to_csv('./global_statistics/task2/wcd_mean.csv',index=False)
    # wcd_median.to_csv('./global_statistics/task2/wcd_median.csv',index=False)

    hcd_mean = pd.read_csv('./global_statistics/task2/hcd_mean.csv')
    hcd_median = pd.read_csv('./global_statistics/task2/hcd_median.csv')
    wcd_mean = pd.read_csv('./global_statistics/task2/wcd_mean.csv')
    wcd_median = pd.read_csv('./global_statistics/task2/wcd_median.csv')

    process_data = process_data.merge(hcd_mean,how='left',on=['hour_min'])
    process_data = process_data.merge(hcd_median,how='left',on=['hour_min'])
    process_data = process_data.merge(wcd_mean,how='left',on=['weekday'])
    process_data = process_data.merge(wcd_median,how='left',on=['weekday'])

    return process_data.drop(['hour_min','weekday'],axis=1)





# ----------------------------------- 处理时间信息 ---------------------------------
def isSpecial(stime):
    time = stime
    if type(time) == str:
        time = pd.to_datetime(time)
    #节假日z
    festival_day = ['2016-10-01', '2016-10-02', '2016-10-03', '2016-10-04', '2016-10-05', '2016-10-06', '2016-10-07']
    #是否工作第一天，而且不是周一
    isFirstWork_day = ['2016-10-08']
    #不是工作第一天但是是周一
    isNotFistWork_day = ['2016-09-19', '2016-10-10']
    #不是工作前一天，而且不是周日
    isBeforeFirstWork_day = ['2016-10-07']
    #不是工作日前一天，而且是周六
    isNotBeforeFirstWork_day = ['2016-10-02', '2016-10-09']
    #是否上班最后一天，而且不是周五
    isEndWork_day = []
    #是否不是上班最后一天，而且是周五
    isNotEndWork_day = ['2016-10-07']
    #是否是放假第一天，而不是周六
    isAfterEndWork_day = []
    #不是放假第一天，但是是周六
    isNotAfterEndWork_day = ['2016-10-08']
    #是否工作日，而且是周末
    isWorkDay = ['2016-10-08', '2016-10-09']
    #不是工作日，但是是周一到周五
    isNotWorkDay = ['2016-10-03', '2016-10-04', '2016-10-05', '2016-10-06', '2016-10-07']
    weekStart = ['2016-09-13', '2016-09-20', '2016-09-27', '2016-10-04', '2016-10-11']
    weekEnd = ['2016-09-19', '2016-09-26', '2016-10-03', '2016-10-10', '2016-10-17']

    month = time.month
    day = time.day
    week = time.weekday()

    month_day = "2016-%02d-%02d" % (month, day)
    # print month_day
    tmp = []

    # 判断是否工作日
    if month_day in isWorkDay:
        tmp.append(1)
    elif month_day in isNotWorkDay:
        tmp.append(0)
    elif week >= 0 and week < 5:
        tmp.append(1)
    else:
        tmp.append(0)

    # 判断是否第一天工作
    if month_day in isFirstWork_day:
        tmp.append(1)
    elif month_day in isNotFistWork_day:
        tmp.append(0)
    elif week == 0:
        tmp.append(1)
    else:
        tmp.append(0)

    # 判断是否工作前一天
    if month_day in isBeforeFirstWork_day:
        tmp.append(1)
    elif month_day in isNotBeforeFirstWork_day:
        tmp.append(0)
    elif week == 6:
        tmp.append(1)
    else:
        tmp.append(0)

    # 判断是否工作最后一天
    if month_day in isEndWork_day:
        tmp.append(1)
    elif month_day in isNotEndWork_day:
        tmp.append(0)
    elif week == 4:
        tmp.append(1)
    else:
        tmp.append(0)

    # 判断是否放假第一天
    if month_day in isAfterEndWork_day:
        tmp.append(1)
    elif month_day in isNotAfterEndWork_day:
        tmp.append(0)
    elif week == 5:
        tmp.append(1)
    else:
        tmp.append(0)

    # 判断是否节假日
    if month_day in festival_day:
        tmp.append(1)
    else:
        tmp.append(0)

    # 判断是否周末
    if week < 5:
        tmp.append(0)
    else:
        tmp.append(1)

    return np.array(tmp)

def processTimeData(process_data):
    process_data['special_day'] = process_data['start_time'].map(lambda x:isSpecial(x))

    process_data['is_WorkDay'] = process_data['special_day'].map(lambda x:x[0])
    process_data['is_FirstWorkDay'] = process_data['special_day'].map(lambda x:x[1])
    process_data['is_BeforeFirstWorkDay'] = process_data['special_day'].map(lambda x:x[2])
    process_data['is_EndWorkDay'] = process_data['special_day'].map(lambda x:x[3])
    process_data['is_AfterEndWorkDay'] = process_data['special_day'].map(lambda x:x[4])
    process_data['is_FestivalDay'] = process_data['special_day'].map(lambda x:x[5])
    process_data['is_WeekEnd'] = process_data['special_day'].map(lambda x:x[6])
    process_data = process_data.drop(['special_day'],axis=1)

    return process_data


# ---------------------------- 处理weather data ----------------------------

def process_weather_data(bin_data):
    # weather_file = './weather/weather_data_bin.csv'
    weather_data = bin_data
    # 填补10月10号的天气缺失数据
    # 10.10号的用10.09和10.11的平均
    date9_weather_data = weather_data[weather_data['date'] == '2016-10-09']
    date11_weather_data = weather_data[weather_data['date'] == '2016-10-11']

    date10_weather_data = date11_weather_data.copy()
    date10_weather_data['date'] = '2016-10-10'
    date10_weather_data['pressure'] = (date9_weather_data['pressure'].values + date11_weather_data[
        'pressure'].values) / 2
    # date10_weather_data['sea_pressure'] = (date9_weather_data['sea_pressure'].values + date11_weather_data[
    #     'sea_pressure'].values) / 2
    # date10_weather_data['wind_direction'] = (date9_weather_data['wind_direction'].values + date11_weather_data[
    #     'wind_direction'].values) / 2
    date10_weather_data['wind_speed'] = (date9_weather_data['wind_speed'].values + date11_weather_data[
        'wind_speed'].values) / 2
    date10_weather_data['temperature'] = (date9_weather_data['temperature'].values + date11_weather_data[
        'temperature'].values) / 2
    date10_weather_data['rel_humidity'] = (date9_weather_data['rel_humidity'].values + date11_weather_data[
        'rel_humidity'].values) / 2
    date10_weather_data['precipitation'] = (date9_weather_data['precipitation'].values + date11_weather_data[
        'precipitation'].values) / 2
    date10_weather_data['wind_direction2'] = (date9_weather_data['precipitation'].values + date11_weather_data[
        'precipitation'].values) / 2
    date10_weather_data['wind_speed2'] = (date9_weather_data['precipitation'].values + date11_weather_data[
        'precipitation'].values) / 2
    date10_weather_data['precipitation2'] = (date9_weather_data['precipitation'].values + date11_weather_data[
        'precipitation'].values) / 2
    date10_weather_data['SSD'] = (date9_weather_data['precipitation'].values + date11_weather_data[
        'precipitation'].values) / 2
    date10_weather_data['SSD_level'] = (date9_weather_data['precipitation'].values + date11_weather_data[
        'precipitation'].values) / 2

    weather_data = pd.concat([weather_data, date10_weather_data], axis=0, ignore_index=True)

    # 整合平均时间与天气数据
    # raw_data['start_time'] = raw_data['time_window'].map(
    #     lambda x: datetime.strptime(x.split(',')[0][1:], '%Y-%m-%d %H:%M:%S'))
    weather_data['date'] = weather_data['date'].astype(str)
    weather_data['hour'] = weather_data['hour'].map(lambda x: ' ' + time(x, 0, 0).strftime('%H:%M:%S'))
    weather_data['start_time'] = weather_data['date'] + weather_data['hour']
    weather_data['start_time'] = weather_data['start_time'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    del weather_data['date']
    del weather_data['hour']
    num = len(weather_data)
    for i in range(num):
        temp = weather_data.loc[i]
        temp1 = copy.deepcopy(temp)
        temp2 = copy.deepcopy(temp)
        temp3 = copy.deepcopy(temp)
        temp4 = copy.deepcopy(temp)
        temp5 = copy.deepcopy(temp)
        temp6 = copy.deepcopy(temp)
        temp7 = copy.deepcopy(temp)
        temp8 = copy.deepcopy(temp)
        stime = temp.start_time
        temp1.start_time = stime + timedelta(minutes=20)
        temp2.start_time = stime + timedelta(minutes=40)
        temp3.start_time = stime + timedelta(minutes=60)
        temp4.start_time = stime + timedelta(minutes=80)
        temp5.start_time = stime + timedelta(minutes=100)
        temp6.start_time = stime + timedelta(minutes=120)
        temp7.start_time = stime + timedelta(minutes=140)
        temp8.start_time = stime + timedelta(minutes=160)
        alltemp = [temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8]
        alltemp = pd.DataFrame(alltemp)
        weather_data = pd.concat([weather_data, alltemp])
    weather_data['start_time'] = weather_data['start_time'].map(str)
    return weather_data


def process_test_weather_data(bin_data):
    # weather_file = './weather/weather_data_bin_test.csv'
    weather_data = bin_data

    weather_data['date'] = weather_data['date'].astype(str)
    weather_data['hour'] = weather_data['hour'].map(lambda x: ' ' + time(x, 0, 0).strftime('%H:%M:%S'))
    weather_data['start_time'] = weather_data['date'] + weather_data['hour']
    weather_data['start_time'] = weather_data['start_time'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    del weather_data['date']
    del weather_data['hour']
    num = len(weather_data)
    for i in range(num):
        temp = weather_data.loc[i]
        temp1 = copy.deepcopy(temp)
        temp2 = copy.deepcopy(temp)
        temp3 = copy.deepcopy(temp)
        temp4 = copy.deepcopy(temp)
        temp5 = copy.deepcopy(temp)
        temp6 = copy.deepcopy(temp)
        temp7 = copy.deepcopy(temp)
        temp8 = copy.deepcopy(temp)
        stime = temp.start_time
        temp1.start_time = stime + timedelta(minutes=20)
        temp2.start_time = stime + timedelta(minutes=40)
        temp3.start_time = stime + timedelta(minutes=60)
        temp4.start_time = stime + timedelta(minutes=80)
        temp5.start_time = stime + timedelta(minutes=100)
        temp6.start_time = stime + timedelta(minutes=120)
        temp7.start_time = stime + timedelta(minutes=140)
        temp8.start_time = stime + timedelta(minutes=160)
        alltemp = [temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8]
        alltemp = pd.DataFrame(alltemp)
        weather_data = pd.concat([weather_data, alltemp])
    weather_data['start_time'] = weather_data['start_time'].map(str)
    # del weather_data['Unnamed: 0']
    return weather_data


# 风向（映射成东风，西风，南风，北风，东北风，东南风，西南风，西北风）
def wind_direction_map(x):
    if 22.5 < x <= 67.5:
        return 1
    elif 67.5 < x <= 112.5:
        return 2
    elif 112.5 < x <= 157.5:
        return 3
    elif 157.5 < x <= 202.5:
        return 4
    elif 202.5 < x <= 247.5:
        return 5
    elif 247.5 < x <= 292.5:
        return 6
    elif 292.5 < x <= 337.5:
        return 7
    else:
        return 8


# 风力，映射为几级风
def wind_speed_map(x):
    if x == 0:
        return 0
    elif 0 < x <= 0.3:
        return 0
    elif 0.3 < x <= 1.6:
        return 1
    elif 1.6 < x <= 3.4:
        return 2
    elif 3.4 < x <= 5.5:
        return 3
    elif 5.5 < x <= 8.0:
        return 4
    elif 8.0 < x <= 10.8:
        return 5
    elif 10.8 < x <= 13.9:
        return 6
    elif 13.9 < x <= 17.2:
        return 7
    elif 17.2 < x <= 20.8:
        return 8
    else:
        return 9


# 雨量等级
def precipitation_map(x):
    if x == 0:
        return 0
    if 0 < x <= 10:
        return 1  # 小雨
    elif 10 < x <= 25:
        return 2  # 中雨
    elif 25 < x <= 50:
        return 3  # 大雨
    elif 50 < x <= 100:
        return 4  # 暴雨
    else:
        return 5


def SSD_map(x):
    if x > 86:
        return 4  # 很热
    if 80 < x <= 86:
        return 3  # 炎热
    elif 76 < x <= 80:
        return 2  # 便热
    elif 71 < x <= 76:
        return 1  # 偏暖
    elif 59 < x <= 71:
        return 0  # 舒适
    elif 51 < x <= 59:
        return -1  # 微凉
    elif 39 < x <= 51:
        return -2  # 清凉
    elif 26 < x <= 39:
        return -3  # 很冷
    else:
        return -4  # 寒冷


def weather_data_bin(path):
    # weather_file = './weather/weather_Oct_18_Oct_24_table7.csv'
    weather_data = pd.read_csv(path)

    des = weather_data.describe()
    # corr_heatmap(weather_data.iloc[:, 2:])

    weather_data['wind_direction2'] = weather_data['wind_direction'].map(wind_direction_map)

    weather_data['wind_speed2'] = weather_data['wind_speed'].map(wind_speed_map)

    weather_data['precipitation2'] = weather_data['precipitation'].map(precipitation_map)

    # 增加人体舒适指数
    weather_data['SSD'] = (1.818 * weather_data['temperature'] + 18.18) * (
            0.88 + 0.002 * weather_data['rel_humidity']) + (weather_data['temperature'] - 32) / (
                                  45 - weather_data['temperature']) - 3.2 * weather_data[
                              'wind_speed'] + 18.2

    weather_data['SSD_level'] = weather_data['SSD'].map(SSD_map)

    del weather_data['sea_pressure']
    del weather_data['wind_direction']
    # weather_data.to_csv('./weather/weather_data_bin_test.csv')
    return weather_data







# ------------------------------------ main -------------------------------------

if __name__ == '__main__':
    is_train = True
    process_base = False # 测试集process_base要设置为true
    input_file, output_file, exp, weather_path = '','','',''
    roll_func = None
    if is_train:
        input_file = './phase1_training/volume_training_phase1_table6.csv'
        output_file = './train/process_data_task2'
        weather_path = './weather/weather_July_01_Oct_17_table7.csv'
    else:
        input_file = './phase1_test/volume_test_phase1_table6.csv'
        output_file = './test/process_data_task2'
        weather_path = './weather/weather_Oct_18_Oct_24_table7.csv'
    if process_base:
        roll_func = pre2hBaseData
        exp = '_base'
    else:
        roll_func = prev2hData
        exp = ''
    output_file += exp + '.csv'
    print('input_file:',input_file)
    print('output_file:',output_file)

    # 读取文件
    df, all_vmodel, all_etc  = readData(input_file,is_train)

    # get rolling data, pre2hBaseData可用于data
    process_data = roll_func(df,all_vmodel,all_etc)

    # process global y feature, remenber to unlock the annotation of code blocks in processGlobal when first run this code
    process_data = processGlobal(process_data)

    # append time features
    process_data = processTimeData(process_data)
    
    # append weather features
    train_weather_bin_data = weather_data_bin(weather_path)
    w_data = process_weather_data(train_weather_bin_data)
    process_data = pd.merge(process_data, w_data, on='start_time', how='left')
    
    # drop自带信息
    process_data = process_data.drop(['vehicle_model_0','vehicle_model_1','vehicle_model_2','vehicle_model_3','vehicle_model_4','vehicle_model_5','vehicle_model_6','vehicle_model_7'\
                                     ,'has_etc_0','has_etc_1'], axis=1)
    if not is_train:
        process_data = process_data.drop(['y'],axis=1)

    process_data.to_csv(output_file,index=False)
    print('processing successfullly, data size:',len(process_data))
    print('Null Size:',process_data.isnull().sum(1).sum())