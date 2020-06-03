#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 20:20:39 2020

@author: liyuan
放在/data5002/的目录下，读取路径如 './phase1_training/trajectories_training_phase1_table5.csv'
"""

import pandas as pd
import numpy as np
import copy
import itertools
from datetime import datetime, timedelta, date, time


def deal_with_table5():
    input_file = './phase1_training/trajectories_training_phase1_table5.csv'
    data = pd.read_csv(input_file)

    result = {col: [] for col in data.columns}
    link = []
    # last_enter_time = []
    enter_time = []
    travel_time = []

    travel_seq = data.travel_seq
    i = 0
    for seq in travel_seq:
        for line in seq.split(';'):
            lk, et, tt = line.split('#')
            link += [lk]
            enter_time += [et]
            travel_time += [tt]
            for col in data.columns:
                result[col].append(data[col][i])
        i += 1
    result['link'] = link
    result['enter_time'] = enter_time
    result['link_travel_time'] = travel_time
    result = pd.DataFrame(result, columns=[*data.columns, 'link', 'enter_time', 'link_travel_time'])
    result = result.drop(['travel_seq'], axis=1)
    output_file = './phase1_training/new_trajectories_train_bypath.csv'  # 存档,处理完travel_seq
    result.to_csv(output_file, index=False)


def getTimeTange(time):
    # 取时间窗口
    left_time = time.replace(minute=(int)(time.minute / 20) * 20, second=0)
    return str(left_time).split(' ')[1]


def miss_data_of_each20min():
    alldata = list(trajectory_train.date.map(str).unique())
    alllink = list(trajectory_train.link.map(str).unique())
    alllink
    dat = []
    for x in itertools.product(alldata, alllink):
        dat.append(x)


def deal_with_new_trajectories_train_bypath():
    output_file = './phase1_training/new_trajectories_train_bypath.csv'
    trajectories_train = pd.read_csv(output_file)
    trajectories_train['starting_time'] = pd.to_datetime(trajectories_train['starting_time'])
    trajectories_train['hour'] = trajectories_train['starting_time'].dt.hour
    trajectories_train['date'] = trajectories_train['starting_time'].dt.date
    trajectories_train['starting_time_20min'] = trajectories_train['starting_time'].map(lambda x: getTimeTange(x))

    # first_2hours = [8,10,17,19]  #后半段
    # first_2hours = [6,8,15,17]  #前半段
    first_2hours = [6, 10, 15, 19]  # 早上4小时，下午4小时
    ind = (trajectories_train['hour'] >= first_2hours[0]) & (trajectories_train['hour'] < first_2hours[1])
    ind1 = (trajectories_train['hour'] >= first_2hours[2]) & (trajectories_train['hour'] < first_2hours[3])
    ind2 = ind | ind1
    trajectory_train = trajectories_train[ind2]
    # 异常值处理
    trajectory_train['link_time_percent'] = trajectory_train['link_travel_time'] / trajectory_train[
        'travel_time']  # 计算比例
    trajectory_train['link_travel_time'][
        trajectory_train['link_travel_time'] > trajectory_train['travel_time']] = np.nan
    # 去掉非时间窗口
    trajectory_train = trajectory_train.drop(
        ['vehicle_id', 'enter_time', 'hour', 'intersection_id', 'tollgate_id', 'starting_time', 'travel_time'], axis=1)
    trajectory_train['start_time'] = trajectory_train['date'].map(str) + " " + trajectory_train[
        'starting_time_20min'].map(str)
    # trajectory_train=trajectory_train.drop(['date','starting_time_20min','link_time_percent'],axis=1)
    trajectory_train['link'] = trajectory_train['link'].map(str)
    trajectory_train['starting_time_20min'] = trajectory_train['start_time'].apply(lambda x: x.split(' ')[1])
    trajectory_mean = trajectory_train.groupby(['link', 'start_time'])['link_travel_time'].mean().reset_index(name='y')

    # 缺失数据
    alldata = list(trajectory_train.date.map(str).unique())
    alllink = list(trajectory_train.link.map(str).unique())
    alltime = list(trajectory_train.starting_time_20min.map(str).unique())
    all_start_time = []
    for x, y in itertools.product(alldata, alltime):
        all_start_time.append(x + ' ' + str(y))

    all_start_time_link = []
    for x in itertools.product(alllink, all_start_time):
        all_start_time_link.append(x)

    trajectory_train = pd.DataFrame(all_start_time_link, columns=['link', 'start_time'])
    trajectory_train = pd.merge(trajectory_train, trajectory_mean, how='left', on=['link', 'start_time'])
    trajectory_train = trajectory_train.fillna(method="ffill")
    return trajectory_train


def get_feature_from_table5():
    output_file = './phase1_training/new_trajectories_train_bypath.csv'
    trajectories_train = pd.read_csv(output_file)
    trajectories_train['starting_time'] = pd.to_datetime(trajectories_train['starting_time'])
    trajectories_train['hour'] = trajectories_train['starting_time'].dt.hour
    trajectories_train['date'] = trajectories_train['starting_time'].dt.date
    trajectories_train['starting_time_20min'] = trajectories_train['starting_time'].map(lambda x: getTimeTange(x))

    # first_2hours = [8,10,17,19]  #后半段
    # first_2hours = [6,8,15,17]  #前半段
    first_2hours = [6, 10, 15, 19]  # 早上4小时，下午4小时
    ind = (trajectories_train['hour'] >= first_2hours[0]) & (trajectories_train['hour'] < first_2hours[1])
    ind1 = (trajectories_train['hour'] >= first_2hours[2]) & (trajectories_train['hour'] < first_2hours[3])
    ind2 = ind | ind1
    trajectory_train = trajectories_train[ind2]
    # 异常值处理
    trajectory_train['link_time_percent'] = trajectory_train['link_travel_time'] / trajectory_train[
        'travel_time']  # 计算比例
    trajectory_train['link_travel_time'][
        trajectory_train['link_travel_time'] > trajectory_train['travel_time']] = np.nan
    link_mean_time = trajectory_train.groupby(['link', 'starting_time_20min'])['link_travel_time'].mean().reset_index(
        name='mean_link_time')
    link_mean_time['link'] = link_mean_time['link'].map(str)
    return link_mean_time


# ----------------------------处理weather data----------------------------
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


# ----------------------------处理link data----------------------------
def process_link_data():
    links_file = './road/links_table3.csv'
    links = pd.read_csv(links_file)
    links_data = links[['link_id', 'length', 'lanes']].rename(columns={'link_id': 'link'})
    links_data['link'] = links_data['link'].map(str)
    return links_data


# ----------------------------处理time data-----------------------------
def createTimeDF():
    df = pd.read_csv('./phase1_training/new_trajectories_train_bypath.csv')

    def getTimeTange(time):
        left_time = time.replace(minute=(int)(time.minute / 20) * 20, second=0)
        return str(left_time)

    def is_holiday(time):
        holiday = ['4-4', '4-5', '4-6', '5-1', '5-2', '5-3', '6-9', '6-10', '6-11', '9-15', '9-16', '9-17', '10-1',
                   '10-2', '10-3', '10-4', '10-5', '10-6', '10-7']
        return (str(time.month) + '-' + str(time.day)) in holiday

    df['stime'] = pd.to_datetime(df['starting_time'])
    df['start_time'] = df['stime'].map(lambda x: getTimeTange(x))
    df['time'] = pd.to_datetime(df['start_time'])
    # df = df.drop(['tollgate_id','direction','vehicle_model','has_etc','vehicle_type'],axis=1)
    # def getTimeTange(time):
    #     left_time = time.replace(minute=(int)(time.minute/20)*20,second=0)
    #     return str(left_time)
    # df['start_time'] = df['time'].map(lambda x:getTimeTange(x))
    df['is_workday'] = (df['time'].dt.weekday < 5) + 0
    df['is_holiday'] = df['time'].map(lambda x: is_holiday(x) + 0)
    wcd = df.groupby(df['time'].dt.weekday).size()
    wcd /= wcd.sum()
    wcd = pd.DataFrame({'weekday': wcd.index, 'weekday_crowd_degree': wcd.values})
    df = df.merge(wcd, left_on=df['time'].dt.weekday, right_on='weekday')
    hcd = df.groupby(df['start_time'].map(lambda x: x.split(' ')[1])).size()
    hcd /= hcd.sum()
    hcd = pd.DataFrame({'hour_min': hcd.index, 'hour_crowd_degree': hcd.values})
    df = df.merge(hcd, left_on=df['start_time'].map(lambda x: x.split(' ')[1]), right_on='hour_min')

    # def compute2hours(time):
    # #     print(time_range)
    #     # end_time = pd.to_datetime(time_range.split(',')[0][1:])
    #     end_time = pd.to_datetime(time)
    #     start_time = end_time + pd.Timedelta(hours=-2)
    # #     print(start_time,end_time)
    #     df2 = df[(df.time>=start_time)&(df.time<end_time)]

    #     cwd = df2.groupby('start_time').size()
    #     time_range = pd.date_range(str(start_time),str(end_time + pd.Timedelta(minutes=-1)),freq='20Min')
    #     for t in time_range:
    #         if str(t) in cwd:
    #             cwd[str(t)]+=0.1
    #         else:
    #             cwd[str(t)]=0.1
    #     cwd /= cwd.sum()
    #     cwd.sort_index()
    #     return tuple(cwd)
    # df['prev2h_crowd_degree'] = df['start_time'].map(lambda x:compute2hours(x))
    # df.drop(['time', 'hour_min','weekday'], axis=1, inplace=True)
    df = df[['time', 'start_time', 'is_workday', 'is_holiday', 'weekday_crowd_degree', 'hour_crowd_degree']].groupby(
        ['time']).max()
    df.to_csv('./phase1_training/time_data.csv', index=False)
    return df


# ------------------------------------ main -------------------------------------

if __name__ == '__main__':
    # deal_with_table5() #这个首次处理完会保存到new_trajectories_train_bypath 之后就不用再跑了，很慢
    # createTimeDF() #根据table5计算拥挤度，保存到phase1_training/time_data.csv

    trajectory_train = deal_with_new_trajectories_train_bypath()  # link总表，link,link_travel_time，start_time
    trajectory_train['link_time'] = trajectory_train['start_time'].apply(lambda x: x.split(' ')[1])

    link_mean_time = get_feature_from_table5()
    link_mean_time = link_mean_time.rename(columns={'starting_time_20min': 'link_time'})

    train_weather_bin_data = weather_data_bin('./weather/weather_July_01_Oct_17_table7.csv')
    w_data = process_weather_data(train_weather_bin_data)
    l_data = process_link_data()
    t_data = pd.read_csv('./phase1_training/time_data.csv')

    # 逐个merge到trajectory_train里
    process_data = pd.merge(trajectory_train, link_mean_time, on=['link', 'link_time'], how='left')
    process_data = pd.merge(process_data, w_data, on='start_time', how='left')
    process_data = pd.merge(process_data, l_data, on='link', how='left')
    process_data = pd.merge(process_data, t_data, on='start_time', how='left')
    # process_data = createTimeDF(process_data)
    # del process_data['Unnamed: 0']
    # process_data.to_csv('./process_data.csv', index=False)

    # --------test data processing---------
    test_weather_bin_data = weather_data_bin('./weather/weather_Oct_18_Oct_24_table7.csv')
    test_weather_data = process_test_weather_data(test_weather_bin_data)
