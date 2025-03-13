# -*- coding: utf-8 -*-
from datetime import datetime, date, timedelta
import math 
import argparse
import subprocess

def save_as_csv(save_dir, df, repartition_num = 10):
    """
    spark dataframe을 csv로 변환후 re-partitioning
    """
    #parquet 버전
    #df.repartition(repartition_num).write.mode("overwrite").parquet(save_dir)
    
    #csv 버전
    df.repartition(repartition_num).write.csv(save_dir, header=True, mode = 'ignore')

def network_df(spark_session, network_path):
    # network(max_speed, linkId 만 담김)
    network_df = spark_session.read.csv(network_path, header=True, inferSchema=True)
    return network_df

def read_parquet(spark_session, parquet_path):
    parquet_sample = spark_session.read.parquet(parquet_path)

    parquet_sample.show()
    parquet_sample.printSchema()

def unix2dt(unix_time):
    return datetime.fromtimestamp(int(unix_time))

def dt2unix(dt):
    if isinstance(dt, date) and not isinstance(dt, datetime):
        dt = datetime(dt.year, dt.month, dt.day)
    epoch = datetime(1970, 1, 1, 9, 0)  # KST = UTC +9h
    delta = dt - epoch
    return int(delta.total_seconds())

def get_date_range(start_ymd, end_ymd):
    try:
        start_date = datetime.strptime(start_ymd, "%Y%m%d").date()
        end_date = datetime.strptime(end_ymd, "%Y%m%d").date()
    except ValueError as e:
        raise ValueError("Probably wrong date inputted: {0}".format(e))
    
    if start_date > end_date:
        raise ValueError("Time is irreversible.")
    
    delta = timedelta(days=1)
    current_date = start_date
    date_list = []
    
    while current_date <= end_date:
        date_list.append(current_date)
        current_date += delta
    
    return date_list

def is_weekend(unix_time):
    # 유닉스 타임스탬프를 datetime 객체로 변환
    dt = datetime.fromtimestamp(int(unix_time))

    # 요일 확인 (월=0, 화=1, ..., 일=6)
    return 1 if dt.weekday() >= 5 else 0 # 토, 일

def calculate_azimuth(x1, y1, x2, y2):
        
        if x1 is None or y1 is None or x2 is None or y2 is None:
            return None  # 또는 math.nan
        
        delta_x = x2 - x1
        delta_y = y2 - y1

        # 라디안 각도
        theta = math.atan2(delta_y, delta_x)
        # 라디안 to degree
        theta_deg = math.degrees(theta)
        # 방위각 변환
        azimuth = (90 - theta_deg) % 360
        
        return azimuth

def check_hdfs_dir_file(hdfs_path):
    """
    특정 hdfs 디렉토리에 파일이 있는 지 검사하는 함수
        returns:
            boolean - 파일이 있으면 True, 없으면 False
    """
    try:
        result = subprocess.run(
            ["hdfs", "dfs", "-ls", hdfs_path],
            capture_output=True, text=True, check=True
        )

        # command 실행 시 Found X items \n log 형태인 지 파악
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            return True
        else:
            return False
        
    # 예외 발생 시 False 처리
    except subprocess.CalledProcessError as e:
        print('error checking HDFS path {} : {}'.format(hdfs_path, e))
        return False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, required=True, choices=['TAXI', 'BUS', 'TRUCK'], help="Choose 'BUS | TRUCK | TAXI'."
    )
    parser.add_argument(
        "--net_level", type=int, required=False, help="Choose '6 | 8', which means LEV6, LEV8 respectively."
    )
    parser.add_argument(
        "--indicator", type=str, choices=['ovrspd', 'ltrm_ovrspd', 'rapid_start', 'rapid_acc', 'rapid_decel', 'rapid_stop', 'rapid_lane_change', 
                                          'rapid_overtake', 'rapid_turn', 'rapid_Uturn',
                                          'link_std_spd','prev_link_std_spd', 'near_link_spd', 'near_link_spd_count'], 
                                          required=True, help="Choose one"
    )
    parser.add_argument(
        "--start_ymd", type=str, required=True, help="Starting date for searching (format: YYYYMMDD)"
    )
    parser.add_argument(
        "--end_ymd", type=str, required=True, help="Ending date for searching (format: YYYYMMDD)"
    )
    parser.add_argument(
        "--timecode", type=int, required=False, default=6, help="Searching at only your interest timecode t. [t-1h, t)" 
    )
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    pass
