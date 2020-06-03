#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 00:59:35 2020

@author: liyuan
"""
import pandas as pd
#import numpy as np
import copy
#import itertools
from datetime import datetime, timedelta, time

def getTimeTange(time):
    # 取时间窗口
    left_time = time.replace(minute=(int)(time.minute / 20) * 20, second=0)
    return str(left_time).split(' ')[1]

def get_leg_2hour(time):
    # 计算每个点距离前两小时的单位
    if int(time.hour)<12:
        left_time = time.replace(hour = 8 ,minute=0, second=0)
    else:
        left_time = time.replace(hour = 17 ,minute=0, second=0)
    leg = (time-left_time).seconds//1200
    return leg







# ----------------------------处理前两小时特征----------------------------


def before_2h_statistics(trajectory):
    trajectory['date'] = trajectory['time'].dt.date
    trajectory['morning'] = trajectory['time'].map(lambda time: 1 if int(time.hour)<12 else 0)
    
    #2h
    trajectory['early2h'] = trajectory['time'].map(lambda time: 1 if (6<=int(time.hour)<8)or(15<=int(time.hour)<17) else 0)
    groupby2h = trajectory[trajectory['early2h']==1].groupby(['link', 'date','morning'])['y']
    linktime_mean_2h = groupby2h.mean().reset_index(name='linktime_mean_2h')
    linktime_median_2h = groupby2h.median().reset_index(name='linktime_median_2h')

    trajectory['early1h'] = trajectory['time'].map(lambda time: 1 if (7<=int(time.hour)<8)or(16<=int(time.hour)<17) else 0)
    linktime_mean_1h = trajectory[trajectory['early1h']==1].groupby(['link', 'date','morning'])['y'].mean().reset_index(name='linktime_mean_1h')
    
    #
    trajectory = pd.merge(trajectory,linktime_mean_2h,how = 'left', on= ['link','date','morning'])
    trajectory = pd.merge(trajectory,linktime_median_2h,how = 'left', on= ['link','date','morning'])    
    trajectory = pd.merge(trajectory,linktime_mean_1h,how = 'left', on= ['link','date','morning'])    
    return trajectory.drop(['date','morning','early2h','early1h'],axis=1)


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

def is_holiday(time):
    holiday = ['4-4', '4-5', '4-6', '5-1', '5-2', '5-3', '6-9', '6-10', '6-11', '9-15', '9-16', '9-17', '10-1',
               '10-2', '10-3', '10-4', '10-5', '10-6', '10-7']
    return (str(time.month) + '-' + str(time.day)) in holiday


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
