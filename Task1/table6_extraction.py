import pandas as pd
from chinese_calendar import is_holiday

def createTimeDF():
    df = pd.read_csv('./phase1_training/volume_training_phase1_table6.csv').sample(1000)
    df['time'] = pd.to_datetime(df['time'])
    df = df.drop(['tollgate_id','direction','vehicle_model','has_etc','vehicle_type'],axis=1)
    def getTimeTange(time):
        left_time = time.replace(minute=(int)(time.minute/20)*20,second=0)
        return str(left_time)
    df['start_time'] = df['time'].map(lambda x:getTimeTange(x))
    df['is_workday'] = (df['time'].dt.weekday<5)+0
    df['is_holiday'] = df['time'].map(lambda x:is_holiday(x)+0)
    wcd  = df.groupby(df['time'].dt.weekday).size()
    wcd /= wcd.sum()
    wcd = pd.DataFrame({'weekday':wcd.index,'weekday_crowd_degree':wcd.values})
    df = df.merge(wcd,left_on=df['time'].dt.weekday,right_on='weekday')
    hcd = df.groupby(df['start_time'].map(lambda x:x.split(' ')[1])).size()
    hcd /= hcd.sum()
    hcd = pd.DataFrame({'hour_min':hcd.index,'hour_crowd_degree':hcd.values})
    df = df.merge(hcd,left_on=df['start_time'].map(lambda x:x.split(' ')[1]),right_on='hour_min')
    def compute2hours(time):
    #     print(time_range)
        # end_time = pd.to_datetime(time_range.split(',')[0][1:])
        end_time = pd.to_datetime(time)
        start_time = end_time + pd.Timedelta(hours=-2)
    #     print(start_time,end_time)
        df2 = df[(df.time>=start_time)&(df.time<end_time)]

        if not df2.size:
            return tuple([1/6 for i in range(6)])
        cwd = df2.groupby('start_time').size()
        time_range = pd.date_range(str(start_time),str(end_time + pd.Timedelta(minutes=-1)),freq='20Min')
        for t in time_range:
            if str(t) in cwd:
                cwd[str(t)]+=0.1
            else:
                cwd[str(t)]=0.1
        cwd /= cwd.sum()
        cwd.sort_index()
        return tuple(cwd)
    df['prev2h_crowd_degree'] = df['start_time'].map(lambda x:compute2hours(x))
    
    return df
df = createTimeDF()


