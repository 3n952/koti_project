# -*- coding: utf-8 -*-
import config as cfg
import utils
import logging

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from pyspark.sql.functions import col, countDistinct, lag, lead, coalesce, lit, max, min, udf, expr, abs, array, explode
from pyspark.sql import functions as F
from pyspark.sql.window import Window

class ElevenMajorDrivingMetrics:
    """
    11대 위험운전 지표 생성을 위해 각 지표 별 판단 기준 적용 클래스

    attributes: 
        spark: spark 세션
        network_df: 제한 속도 정보가 담긴 표준노드링크 dataframe
        args: 사용자 입력 arguments -> dict
    """
    
    def __init__(self, spark, network_df, args):
        self.spark = spark
        self.network_df = network_df
        self.args = args
        self.df = None
    
    def ps_load_rdd(self, rdd_file_path):
        """
        ps 데이터의 rdd 파일 파싱 후 spark dataFrame변환 및 1차 가공
            
            args: 
                rdd_file: rdd 파일 경로
        """
        
        def ps_parse_line(iterator):
            """
            로드된 rdd 파일 파싱
                args:
                    iterator: rdd 파일 내 한 줄 텍스트
                returns:
                    파싱된 내용 리스트
            """
            ps_data = []
            for line in iterator:
                line_list = line.strip().split(',')
                fixed_data = line_list[:8]
                repeat_line_list = line_list[8:]
                repeat_dataset = [repeat_line_list[i:i + 13] for i in range(0, len(repeat_line_list), 13)]
                
                for grp in repeat_dataset:
                    row_data = fixed_data + grp
                    temp = []
                    uti_data = []  # Unique_Trip_Id: OBU_ID + GRP_ID + vehType
                    for col_idx, data in enumerate(row_data):
                        if col_idx in [0, 1, 2, 6, 8, 9, 10, 11, 12, 14, 18]:
                            if col_idx in [0, 1, 2]:
                                uti_data.append(data)
                                if col_idx == 2:
                                    uti_data = '_'.join(map(str, uti_data))
                                    temp.append(uti_data)
                                continue
                            temp.append(data)
                        
                    ps_data.append(','.join(temp))
    
            return ps_data
        
        schema = StructType([
        StructField("TripID", StringType(), True),
        StructField("matchingRate", FloatType(), True),
        StructField("pointTime", IntegerType(), True),
        StructField("pointX", FloatType(), True),
        StructField("pointY", FloatType(), True),
        StructField("spd", FloatType(), True),
        StructField("acc", FloatType(), True),
        StructField("vLinkID", StringType(), True),
        StructField('gisSpeed', FloatType(), True)])

        rdd = self.spark.sparkContext.textFile(rdd_file_path)
        parsed_rdd = rdd.mapPartitions(ps_parse_line)
        self.df = self.spark.read.schema(schema).csv(parsed_rdd)

    def make_final_df(self):
        """
        1차 가공된 dataFrame의 data에서 이상치를 제거하고 목적에 맞게 전처리
        """
        # DTG 데이터 이상치 제거 + 신뢰도(matchingRate) 확보
        self.df = self.df.filter(
            (self.df['vLinkID'] != "-1") & (self.df['matchingRate'] >= 90)
        )
        self.df = self.df.drop('matchingRate')
        self.df = self.df.filter(col('gisSpeed').cast('float') != -1.0)
        self.df = self.df.filter(~((col("pointX") == 0.0) | (col("pointY") == 0.0)))
        self.df = self.df.drop('acc')

        window_spec = Window.partitionBy('TripId').orderBy('pointTime')

        # ps data filtering
        self.df = self.df.join(self.network_df, on="vLinkId", how="inner") # 기존 DTG dataframe에 network의 max_speed 열 추가하기
        self.df = self.df.filter((col('max_speed').cast('float') != 0.0) | (col('max_speed').cast('float') != -1.0))
        self.df = self.df.withColumn('acc', coalesce(col('spd') - lag('spd').over(window_spec), lit(0.0)))
        self.df = self.df.withColumn('spd', col('spd') * 3.6) # km/h변환
        self.df = self.df.withColumn('acc', col('acc') * 12.96) # km/h**2 변환

    def ovrspd_metric(self):
        """
        1차 가공 DTG df로부터 과속 여부를 판단하기 위한 기준으로 필터링한 df를 만들기 위한 함수
            returns:
                ovrspd_df: 링크별 과속 트립 대수 dataFrame
        """
        n_df = self.df.groupBy('vLinkId').agg(countDistinct('TripId').alias('n'))

        # 과속 여부 필터링
        basic_ovrspd_df = self.df.filter(col('gisSpeed').cast('float') > col('max_speed').cast('float') + 20.0)
        ovrspd_df = basic_ovrspd_df.groupBy('vLinkId').agg(countDistinct('TripID').alias('ovrspd'))
        ovrspd_df = ovrspd_df.join(n_df, on='vLinkId', how='inner')
        
        return ovrspd_df # spark DataFrame
    
    def ltrm_ovrspd_metric(self):
        """
        1차 가공 DTG df로부터 장기과속 여부를 판단하기 위한 기준으로 필터링한 df를 만들기 위한 함수
            returns:
                ltrm_ovrspd_df2: 링크별 장기 과속 트립 대수 dataFrame
        """
        n_df = self.df.groupBy('vLinkId').agg(countDistinct('TripId').alias('n'))
        basic_ovrspd_df = self.df.filter(col('gisSpeed').cast('float') > col('max_speed').cast('float') + 20.0)
        ltrm_ovrspd_df = basic_ovrspd_df.groupBy('vLinkId', 'TripId').agg(max('pointTime').alias('max_t'), min('pointTime').alias('min_t'))
        ltrm_ovrspd_df = ltrm_ovrspd_df.filter((col('max_t').cast('float') - col('min_t').cast('float')) >= 180.0)

        #ltrm_ovrspd 갯수만 컬럼으로 남김
        ltrm_ovrspd_df = ltrm_ovrspd_df.groupBy('vLinkId').agg(countDistinct('TripID').alias('ltrm_ovrspd'))

        ltrm_ovrspd_df = ltrm_ovrspd_df.join(n_df, on='vLinkId', how='inner')

        return ltrm_ovrspd_df

    def rapid_accel_metric(self):
        """
        1차 가공 DTG df로부터 급가속 여부를 판단하기 위한 기준으로 필터링한 df를 만들기 위한 함수
            returns:
                rapid_acc_df: 링크별 급가속 트립 대수 dataFrame
        """
        n_df = self.df.groupBy('vLinkId').agg(countDistinct('TripId').alias('n'))
        rapid_acc_df = self.df.filter(col('spd').cast('float') >= 6.0)

        if self.args.mode == 'TAXI':
            rapid_acc_df1 = rapid_acc_df.filter((col('spd').cast('float').between(6.0, 10.0)) & (col('acc').cast('float') >= 12.0))
            rapid_acc_df2 = rapid_acc_df.filter((col('spd').cast('float').between(10.0, 20.0)) & (col('acc').cast('float') >= 10.0))
            rapid_acc_df3 = rapid_acc_df.filter((col('spd').cast('float') > 20.0) & (col('acc').cast('float') >= 8.0))
        
        elif self.args.mode == 'BUS':
            rapid_acc_df1 = rapid_acc_df.filter((col('spd').cast('float').between(6.0, 10.0)) & (col('acc').cast('float') >= 8.0))
            rapid_acc_df2 = rapid_acc_df.filter((col('spd').cast('float').between(10.0, 20.0)) & (col('acc').cast('float') >= 7.0))
            rapid_acc_df3 = rapid_acc_df.filter((col('spd').cast('float') > 20.0) & (col('acc').cast('float') >= 6.0))

        else:
            rapid_acc_df1 = rapid_acc_df.filter((col('spd').cast('float').between(6.0, 10.0)) & (col('acc').cast('float') >= 7.0))
            rapid_acc_df2 = rapid_acc_df.filter((col('spd').cast('float').between(10.0, 20.0)) & (col('acc').cast('float') >= 6.0))
            rapid_acc_df3 = rapid_acc_df.filter((col('spd').cast('float') > 20.0) & (col('acc').cast('float') >= 5.0))
        
        rapid_acc_df = rapid_acc_df1.union(rapid_acc_df2).union(rapid_acc_df3)
        rapid_acc_df = rapid_acc_df.groupBy('vLinkId').agg(countDistinct('TripID').alias('rapid_acc'))

        
        rapid_acc_df = rapid_acc_df.join(n_df, on='vLinkId', how='inner')
        return rapid_acc_df
        
    def rapid_start_metric(self):
        """
        1차 가공 DTG df로부터 급출발 여부를 판단하기 위한 기준으로 필터링한 df를 만들기 위한 함수
            returns:
                rapid_start_df: 링크별 급출발 트립 대수 dataFrame
        """
        n_df = self.df.groupBy('vLinkId').agg(countDistinct('TripId').alias('n'))
        rapid_start_df = self.df.filter(col('spd').cast('float') <= 5.0)

        if self.args.mode == 'TAXI':
            rapid_start_df = rapid_start_df.filter(col('acc').cast('float') >= 10.0)

        elif self.args.mode == 'BUS':
            rapid_start_df = rapid_start_df.filter(col('acc').cast('float') >= 8.0)

        else:
            rapid_start_df = rapid_start_df.filter(col('acc').cast('float') >= 6.0)
        
        rapid_start_df = rapid_start_df.groupBy('vLinkId').agg(countDistinct('TripID').alias('rapid_start'))
        rapid_start_df = rapid_start_df.join(n_df, on='vLinkId', how='inner')
        return rapid_start_df
    
    def rapid_decel_metric(self):
        """
        1차 가공 DTG df로부터 급감속 여부를 판단하기 위한 기준으로 필터링한 df를 만들기 위한 함수
            returns:
                rapid_decel_df: 링크별 급감속 트립 대수 dataFrame
        """
        n_df = self.df.groupBy('vLinkId').agg(countDistinct('TripId').alias('n'))
        if self.args.mode == 'TAXI':
            rapid_decel_df1 = self.df.filter((col('spd').cast('float').between(6.0, 30.0)) & (col('acc').cast('float') <= -14.0))
            rapid_decel_df2 = self.df.filter((col('spd').cast('float').between(6.0, 50.0)) & (col('acc').cast('float') <= -15.0))
            rapid_decel_df3 = self.df.filter((col('spd').cast('float') > 50.0) & (col('acc').cast('float') <= -15.0))
            rapid_decel_df = rapid_decel_df1.union(rapid_decel_df2).union(rapid_decel_df3)
            rapid_decel_df = rapid_decel_df.groupBy('vLinkId').agg(countDistinct('TripID').alias('rapid_decel'))

        elif self.args.mode == 'BUS':
            rapid_decel_df1 = self.df.filter((col('spd').cast('float').between(6.0, 30.0)) & (col('acc').cast('float') <= -9.0))
            rapid_decel_df2 = self.df.filter((col('spd').cast('float').between(6.0, 50.0)) & (col('acc').cast('float') <= -10.0))
            rapid_decel_df3 = self.df.filter((col('spd').cast('float') > 50.0) & (col('acc').cast('float') <= -12.0))
            rapid_decel_df = rapid_decel_df1.union(rapid_decel_df2).union(rapid_decel_df3)
            rapid_decel_df = rapid_decel_df.groupBy('vLinkId').agg(countDistinct('TripID').alias('rapid_decel'))

        else:
            rapid_decel_df = self.df.filter((col('spd').cast('float') >= 6.0) & (col('acc').cast('float') <= -8.0))
            rapid_decel_df = rapid_decel_df.groupBy('vLinkId').agg(countDistinct('TripID').alias('rapid_decel'))
        rapid_decel_df = rapid_decel_df.join(n_df, on='vLinkId', how='inner')
        return rapid_decel_df

    def rapid_stop_metric(self):
        """
        1차 가공 DTG df로부터 급정지 여부를 판단하기 위한 기준으로 필터링한 df를 만들기 위한 함수
            returns:
                rapid_stop_df: 링크별 급정지 트립 대수 dataFrame
        """
        n_df = self.df.groupBy('vLinkId').agg(countDistinct('TripId').alias('n'))
        if self.args.mode == 'TAXI':
            rapid_stop_df = self.df.filter((col('acc').cast('float') <= -14.0) & (col('spd').cast('float') <= 5.0))
            rapid_stop_df = rapid_stop_df.groupBy('vLinkId').agg(countDistinct('TripID').alias('rapid_stop'))


        elif self.args.mode == 'BUS':
            rapid_stop_df = self.df.filter((col('acc').cast('float') <= -9.0) & (col('spd').cast('float') <= 5.0))
            rapid_stop_df = rapid_stop_df.groupBy('vLinkId').agg(countDistinct('TripID').alias('rapid_stop'))

        else:
            rapid_stop_df = self.df.filter((col('acc').cast('float') <= -8.0) & (col('spd').cast('float') <= 5.0))
            rapid_stop_df = rapid_stop_df.groupBy('vLinkId').agg(countDistinct('TripID').alias('rapid_stop'))
        rapid_stop_df = rapid_stop_df.join(n_df, on='vLinkId', how='inner')
        return rapid_stop_df
    
    def rapid_lane_change_metric(self):
        """
        1차 가공 DTG df로부터 급진로변경 여부를 판단하기 위한 기준으로 필터링한 df를 만들기 위한 함수
            returns:
                rapid_lane_change_df: 링크별 급진로변경 트립 대수 dataFrame
        """
        from utils import calculate_azimuth
        calculate_azimuth = udf(calculate_azimuth, FloatType())
        n_df = self.df.groupBy('vLinkId').agg(countDistinct('TripId').alias('n'))
              
        rapid_lane_change_df = self.df.filter(abs(col('acc').cast('float')) <= 2.0)
        window_spec = Window.partitionBy('TripId').orderBy('pointTime')

        # 새 컬럼에 동일 트립 내 이전 포인트 시간의 위도, 경도 좌표 추가
        rapid_lane_change_df = rapid_lane_change_df.withColumn("next_pointX", lead("pointX").over(window_spec))
        rapid_lane_change_df = rapid_lane_change_df.withColumn("next_pointY", lead("pointY").over(window_spec))


        rapid_lane_change_df = rapid_lane_change_df.withColumn("azimuth", calculate_azimuth( \
                                                            col("pointX"), col("pointY"), col("next_pointX"), col("next_pointY")))
            
        rapid_lane_change_df = rapid_lane_change_df.withColumn('degree', (((lead('azimuth').over(window_spec) - col('azimuth'))
                                                                   + 180) % 360) -180)
        if self.args.mode == 'TAXI':                                
            rapid_lane_change_df = rapid_lane_change_df.filter(abs(col('degree')) >= 10.0)

        elif self.args.mode == 'BUS':
            rapid_lane_change_df = rapid_lane_change_df.filter(abs(col('degree')) >= 8.0)

        elif self.args.mode == 'TRUCK':
            rapid_lane_change_df = rapid_lane_change_df.filter(abs(col('degree')) >= 6.0)
        
        # 누적각도 - 5초 동안의 각도 합
        rapid_lane_change_df = rapid_lane_change_df.withColumn('degree_agg', array(*[lag('degree', i).over(window_spec) for i in range(1,6)]))
        rapid_lane_change_df = rapid_lane_change_df.withColumn('degree_agg_sum', expr("aggregate(degree_agg, cast(0.0 as double), (acc, x) -> acc + x)"))
        rapid_lane_change_df = rapid_lane_change_df.filter((col('degree_agg_sum').isNotNull()) & (abs(col('degree_agg_sum')) <= 2.0))
        rapid_lane_change_df = rapid_lane_change_df.groupBy('vLinkID').agg(countDistinct('TripId').alias('rapid_lane_change'))

        rapid_lane_change_df = rapid_lane_change_df.join(n_df, on='vLinkId', how='inner')
        return rapid_lane_change_df

    def rapid_overtake_metric(self):
        """
        1차 가공 DTG df로부터 급진로변경 여부를 판단하기 위한 기준으로 필터링한 df를 만들기 위한 함수
            returns:
                rapid_overtake_df: 링크별 급앞지르기 트립 대수 dataFrame
        """
        from utils import calculate_azimuth
        calculate_azimuth = udf(calculate_azimuth, FloatType())
        n_df = self.df.groupBy('vLinkId').agg(countDistinct('TripId').alias('n'))

        rapid_overtake_df = self.df.filter(col('acc').cast('float') >= 3.0)
        rapid_overtake_df = rapid_overtake_df.filter(col('spd').cast('float') >= 30.0)
        window_spec = Window.partitionBy('TripId').orderBy('pointTime')

         # 새 컬럼에 동일 트립 내 이전 포인트 시간의 위도, 경도 좌표 추가
        rapid_overtake_df = rapid_overtake_df.withColumn("next_pointX", lead("pointX").over(window_spec))
        rapid_overtake_df = rapid_overtake_df.withColumn("next_pointY", lead("pointY").over(window_spec))
        rapid_overtake_df = rapid_overtake_df.withColumn("azimuth", calculate_azimuth( \
                                                            col("pointX"), col("pointY"), col("next_pointX"), col("next_pointY")))
        rapid_overtake_df = rapid_overtake_df.withColumn('degree', (((lead('azimuth').over(window_spec) - col('azimuth'))
                                                                   + 180) % 360) -180)
        
        if self.args.mode == 'TAXI':
            rapid_overtake_df = rapid_overtake_df.filter(abs(col('degree')) >= 10.0) 

        elif self.args.mode == 'BUS':
            rapid_overtake_df = rapid_overtake_df.filter(abs(col('degree')) >= 8.0)
        
        elif self.args.mode == 'TRUCK':
            rapid_overtake_df = rapid_overtake_df.filter(abs(col('degree')) >= 6.0)
            

        rapid_overtake_df = rapid_overtake_df.withColumn('degree_agg', array(*[lag('degree', i).over(window_spec) for i in range(1,6)]))            
        rapid_overtake_df = rapid_overtake_df.withColumn('degree_agg_sum',  expr("aggregate(degree_agg, cast(0.0 as double), (acc, x) -> acc + x)"))
        rapid_overtake_df = rapid_overtake_df.filter((col('degree_agg_sum').isNotNull()) & (abs(col('degree_agg_sum')) <= 2.0))
        rapid_overtake_df = rapid_overtake_df.groupBy('vLinkID' ).agg(countDistinct('TripId').alias('rapid_overtake'))

        rapid_overtake_df = rapid_overtake_df.join(n_df, on='vLinkId', how='inner')
        return rapid_overtake_df

    def rapid_turn_metric(self):
        """
        1차 가공 DTG df로부터 급회전 여부를 판단하기 위한 기준으로 필터링한 df를 만들기 위한 함수
            returns:
                rapid_overtake_df: 링크별 급앞지르기 트립 대수 dataFrame
        """
        from utils import calculate_azimuth
        calculate_azimuth = udf(calculate_azimuth, FloatType())
        window_spec = Window.partitionBy('TripId').orderBy('pointTime')
        n_df = self.df.groupBy('vLinkId').agg(countDistinct('TripId').alias('n'))
        
        if self.args.mode == 'TAXI':
            rapid_turn_df = self.df.filter(col('spd').cast('float') >= 30.0)
            rapid_turn_df = rapid_turn_df.withColumn("next_pointX", lead("pointX").over(window_spec))
            rapid_turn_df = rapid_turn_df.withColumn("next_pointY", lead("pointY").over(window_spec))
            rapid_turn_df = rapid_turn_df.withColumn("azimuth", calculate_azimuth( \
                                                            col("pointX"), col("pointY"), col("next_pointX"), col("next_pointY")))
            rapid_turn_df = rapid_turn_df.withColumn('degree', (((lead('azimuth').over(window_spec) - col('azimuth'))
                                                                   + 180) % 360) -180)
            
            rapid_turn_df = rapid_turn_df.withColumn('degree_agg', array(*[lead('degree', i).over(window_spec) for i in range(1,3)]))
            

        elif self.args.mode == 'BUS':
            rapid_turn_df = self.df.filter(col('spd').cast('float') >= 25.0)
            rapid_turn_df = rapid_turn_df.withColumn("next_pointX", lead("pointX").over(window_spec))
            rapid_turn_df = rapid_turn_df.withColumn("next_pointY", lead("pointY").over(window_spec))
            rapid_turn_df = rapid_turn_df.withColumn("azimuth", calculate_azimuth( \
                                                            col("pointX"), col("pointY"), col("next_pointX"), col("next_pointY")))
            rapid_turn_df = rapid_turn_df.withColumn('degree', (((lead('azimuth').over(window_spec) - col('azimuth'))
                                                                   + 180) % 360) -180)
            
            rapid_turn_df = rapid_turn_df.withColumn('degree_agg', array(*[lead('degree', i).over(window_spec) for i in range(1,4)]))
        
        elif self.args.mode == 'TRUCK':
            rapid_turn_df = self.df.filter(col('spd').cast('float') >= 20.0)
            rapid_turn_df = rapid_turn_df.withColumn("next_pointX", lead("pointX").over(window_spec))
            rapid_turn_df = rapid_turn_df.withColumn("next_pointY", lead("pointY").over(window_spec))
            rapid_turn_df = rapid_turn_df.withColumn("azimuth", calculate_azimuth( \
                                                            col("pointX"), col("pointY"), col("next_pointX"), col("next_pointY")))
            rapid_turn_df = rapid_turn_df.withColumn('degree', (((lead('azimuth').over(window_spec) - col('azimuth'))
                                                                   + 180) % 360) -180)
            
            rapid_turn_df = rapid_turn_df.withColumn('degree_agg', array(*[lead('degree', i).over(window_spec) for i in range(1,4)]))
        
        rapid_turn_df = rapid_turn_df.withColumn("degree_agg_exploded", explode(col("degree_agg")))
        rapid_turn_df = rapid_turn_df.filter((abs(col("degree_agg_exploded")) >= 60) & (abs(col("degree_agg_exploded")) <= 160))
        rapid_turn_df = rapid_turn_df.groupBy('vLinkID' ).agg(countDistinct('TripId').alias('rapid_turn'))

        rapid_turn_df = rapid_turn_df.join(n_df, on='vLinkId', how='inner')
        return rapid_turn_df

    def rapid_Uturn_metric(self):
        """
        1차 가공 DTG df로부터 급U턴 여부를 판단하기 위한 기준으로 필터링한 df를 만들기 위한 함수
            returns:
                rapid_overtake_df: 링크별 급앞지르기 트립 대수 dataFrame
        """
        from utils import calculate_azimuth
        calculate_azimuth = udf(calculate_azimuth, FloatType())
        n_df = self.df.groupBy('vLinkId').agg(countDistinct('TripId').alias('n'))
        window_spec = Window.partitionBy('TripId').orderBy('pointTime')

        if self.args.mode == 'TAXI':
            rapid_Uturn = self.df.filter(col('spd').cast('float') >= 25.0)
            rapid_Uturn = rapid_Uturn.withColumn("next_pointX", lead("pointX").over(window_spec))
            rapid_Uturn = rapid_Uturn.withColumn("next_pointY", lead("pointY").over(window_spec))
            rapid_Uturn = rapid_Uturn.withColumn("azimuth", calculate_azimuth( \
                                                            col("pointX"), col("pointY"), col("next_pointX"), col("next_pointY")))
            rapid_Uturn = rapid_Uturn.withColumn('degree', (((lead('azimuth').over(window_spec) - col('azimuth'))
                                                                   + 180) % 360) -180)
            
            rapid_Uturn = rapid_Uturn.withColumn('degree_agg', array(*[lead('degree', i).over(window_spec) for i in range(1,5)]))

        elif self.args.mode == 'BUS':
            rapid_Uturn = self.df.filter(col('spd').cast('float') >= 20.0)
            rapid_Uturn = rapid_Uturn.withColumn("next_pointX", lead("pointX").over(window_spec))
            rapid_Uturn = rapid_Uturn.withColumn("next_pointY", lead("pointY").over(window_spec))
            rapid_Uturn = rapid_Uturn.withColumn("azimuth", calculate_azimuth( \
                                                            col("pointX"), col("pointY"), col("next_pointX"), col("next_pointY")))
            rapid_Uturn = rapid_Uturn.withColumn('degree', (((lead('azimuth').over(window_spec) - col('azimuth'))
                                                                   + 180) % 360) -180)
            rapid_Uturn = rapid_Uturn.withColumn('degree_agg', array(*[lead('degree', i).over(window_spec) for i in range(1,7)]))
    
        elif self.args.mode == 'TRUCK':
            rapid_Uturn = self.df.filter(col('spd').cast('float') >= 15.0)
            rapid_Uturn = rapid_Uturn.withColumn("next_pointX", lead("pointX").over(window_spec))
            rapid_Uturn = rapid_Uturn.withColumn("next_pointY", lead("pointY").over(window_spec))
            rapid_Uturn = rapid_Uturn.withColumn("azimuth", calculate_azimuth( \
                                                            col("pointX"), col("pointY"), col("next_pointX"), col("next_pointY")))
            rapid_Uturn = rapid_Uturn.withColumn('degree', (((lead('azimuth').over(window_spec) - col('azimuth'))
                                                                   + 180) % 360) -180)
            
            rapid_Uturn = rapid_Uturn.withColumn('degree_agg', array(*[lead('degree', i).over(window_spec) for i in range(1,7)]))

        rapid_Uturn = rapid_Uturn.withColumn("degree_agg_exploded", explode(col("degree_agg")))
        rapid_Uturn = rapid_Uturn.filter((abs(col("degree_agg_exploded")) >= 60) & (abs(col("degree_agg_exploded")) <= 160))
        rapid_Uturn = rapid_Uturn.groupBy('vLinkID' ).agg(countDistinct('TripId').alias('rapid_Uturn'))

        rapid_Uturn = rapid_Uturn.join(n_df, on='vLinkId', how='inner')
        return rapid_Uturn

def main(args, spark, logger):
    '''
    각종 지표를 취합하여 설계된 데이터 테이블을 산출하고 이를 hdfs으로 저장하는 메인 함수
        args:
            args: 사용자 정의 인자 값
            spark: 스파크 세션
            logger: logger 객체
    '''

    mode = args.mode.upper()
    net_level = args.net_level
    network_df = utils.network_df(spark, cfg.NETWORK_CSV)
    indicator_metric = ElevenMajorDrivingMetrics(spark, network_df, args)
    date_range = utils.get_date_range(args.start_ymd, args.end_ymd)
    
    result_schema = StructType([
            StructField("vLinkId", StringType(), True),
            StructField(args.indicator, IntegerType(), True),
            StructField("total_traffic", IntegerType(), True)
        ])
    
    result_df = spark.createDataFrame([], result_schema)

    for idx, ymd in enumerate(date_range):
        logger.info('Progress: {}/{}'.format(idx+1, len(date_range)))

        y, m, d = ymd.strftime("%Y_%m_%d").split('_')
        if net_level is not None:
            input_path = "hdfs:/data/DTG/PS_{}_LEV{}/{}/{}/{}/*".format(mode, net_level, y, m, d)
        else:
            input_path = "hdfs:/data/DTG/PS_{}_MOCT_NEW/{}/{}/{}/*".format(mode, y, m, d)

        indicator_metric.ps_load_rdd(input_path)
        indicator_metric.make_final_df()

        if args.indicator == 'ovrspd':
            indicator_df = indicator_metric.ovrspd_metric()
        elif args.indicator == 'ltrm_ovrspd':
            indicator_df = indicator_metric.ltrm_ovrspd_metric()
        elif args.indicator == 'rapid_acc':
            indicator_df = indicator_metric.rapid_accel_metric()
        elif args.indicator == 'rapid_start':
            indicator_df = indicator_metric.rapid_start_metric()
        elif args.indicator == 'rapid_decel':
            indicator_df = indicator_metric.rapid_decel_metric()
        elif args.indicator == 'rapid_stop':
            indicator_df = indicator_metric.rapid_stop_metric()
        elif args.indicator == 'rapid_lane_change':
            indicator_df = indicator_metric.rapid_lane_change_metric()
        elif args.indicator == 'rapid_overtake':
            indicator_df = indicator_metric.rapid_overtake_metric()
        elif args.indicator == 'rapid_turn':
            indicator_df = indicator_metric.rapid_turn_metric()
        elif args.indicator == 'rapid_Uturn':
            indicator_df = indicator_metric.rapid_Uturn_metric()
        else:
            print('define indicator what to calculate')
        
        result_df = result_df.union(indicator_df)
        result_df.show(3)

    result_df = result_df.groupBy('vLinkID' ).agg(F.sum(col(args.indicator)).alias(args.indicator), 
                                                  F.sum('total_traffic').alias('total_traffic'))

    # hdfs 저장 경로
    output_file = '/{}/{}'.format(mode, args.indicator)
    logger.info('{}_{} indicator saved!'.format(mode, args.indicator))
    output_path = cfg.OUTPUT_DIR + output_file
    output_rdd = result_df.rdd.map(lambda row: ",".join([str(item) for item in row]))
    output_rdd.coalesce(10).saveAsTextFile(output_path)

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.info('start spark session')

    args = utils.get_args()
    APPLICATION_NAME = cfg.APPLICATION_NAME[0]+"_{}_{}(time range: {}~{})".format(args.mode, args.indicator, args.start_ymd, args.end_ymd)

    # session build
    spark = SparkSession.builder.appName(APPLICATION_NAME) \
    .config("spark.executor.instances", "10") \
    .config("spark.executor.memory", "15g") \
    .config("spark.executor.cores", '5') \
    .getOrCreate()

    # PySpark 로그 레벨 낮추기 (INFO 메시지 제거)
    spark.sparkContext.setLogLevel("WARN")
    # Pyspark UDF 관련 참조 문제 해결 - addpyfile
    spark.sparkContext.addPyFile("/home/hadoop/odyssey/san/utils.py")

    # 지표 산출 
    main(args, spark, logger)

    # session exit
    spark.stop()

