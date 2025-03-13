# -*- coding: utf-8 -*-
import config as cfg, utils, logging
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, IntegerType
from pyspark.sql.functions import col, countDistinct, from_unixtime, hour, udf, sum, lag, lead, when, abs, first, sqrt
from pyspark.sql import functions as F
from pyspark.sql.window import Window

class SpeedSafetyMetrics:
    """
    속도 기반 안전 지표 생성을 위해 각 지표 별 판단 기준 적용 클래스

    attributes: 
        spark: spark 세션
        args: 사용자 입력 arguments -> dict
    """

    def __init__(self, spark, args):
        self.spark = spark
        self.args = args
        self.df = None

    def p1_load_rdd(self, rdd_file_path):
        """
        p1 데이터의 rdd 파일 파싱 후 spark dataFrame변환 및 1차 가공
        """

        def p1_parse_line(iterator):
            """
            로드된 rdd 파일 파싱
                args:
                    iterator: rdd 파일 내 한 줄 텍스트
                returns:
                    파싱된 내용 리스트
            """
            p1_data = []
            for line in iterator:
                line_list = line.strip().split(',')
                fixed_data = line_list[:8]
                repeat_line_list = line_list[8:]
                repeat_dataset = [repeat_line_list[i:i + 7] for i in range(0, len(repeat_line_list), 7)]
                for grp in repeat_dataset:
                    row_data = fixed_data + grp
                    temp = []
                    uti_data = []
                    for col_idx, data in enumerate(row_data):
                        if col_idx in (0, 1, 2, 6, 8, 9, 10, 12):
                            if col_idx in (0, 1, 2):
                                uti_data.append(data)
                                if col_idx == 2:
                                    uti_data = ('_').join(map(str, uti_data))
                                    temp.append(uti_data)
                                continue
                            temp.append(data)
                    p1_data.append((',').join(temp))

            return p1_data

        schema = StructType([
         StructField('TripID', StringType(), True),
         StructField('matchingRate', FloatType(), True),
         StructField('vLinkId', StringType(), True),
         StructField('inTime', FloatType(), True),
         StructField('outTime', FloatType(), True),
         StructField('gisSpeed', FloatType(), True)])
        rdd = self.spark.sparkContext.textFile(rdd_file_path)
        parsed_rdd = rdd.mapPartitions(p1_parse_line)
        self.df = self.spark.read.schema(schema).csv(parsed_rdd)
        # print('distinct link count')
        # self.df.select(countDistinct('vLinkId')).show()

    def make_final_df(self):
        """
        1차 가공된 dataFrame의 data에서 이상치를 제거하고 목적에 맞게 전처리
        """
        from utils import is_weekend
        is_weekend_udf = udf(is_weekend, IntegerType())
        self.df = self.df.filter((self.df['vLinkID'] != '-1') & (self.df['matchingRate'] >= 90))
        self.df = self.df.drop('matchingRate')
        self.df = self.df.filter(col('gisSpeed').cast('float') != -1.0)
        self.df = self.df.filter(col('inTime').cast('float') < col('outTime').cast('float'))
        self.df = self.df.withColumn('inHour', hour(from_unixtime(col('inTime'), 'yyyy-MM-dd HH:mm:ss')))\
                         .withColumn('outHour', hour(from_unixtime(col('outTime'), 'yyyy-MM-dd HH:mm:ss')))\
                         .withColumn('weekendCode', is_weekend_udf(col('inTime')))

    def link_spd_std(self, timecoderange):
        """
        1차 가공 DTG df로부터 링크 별 속도 표준편차 df 생성
            returns:
                link_spd_std_df: 링크별 통행량, 통행속도 총합의 속성정보가 담긴 df
        """
        #self.df = self.df.filter((col('inHour') == timecoderange - 1) & (col('outHour') == timecoderange - 1))
        self.df = self.df.filter(col('inHour') == timecoderange - 1)

        n_df = self.df.groupBy('vLinkId').agg(countDistinct('TripId').alias('n'))
        x_bar_df = self.df.groupBy('vLinkId').agg(sum('gisSpeed').alias('x_sum_per_link'))
        link_spd_std_df = self.df.join(n_df, on='vLinkId', how='left').join(x_bar_df, on='vLinkId', how='left')
        link_spd_std_df = link_spd_std_df.withColumnRenamed('n', 'total_traffic')
        return link_spd_std_df

    def near_link_spd_count(self, timecoderange):
        """
        1차 가공 DTG df로부터 인접 링크 간 주행속도 차이를 속도 구간별로 나눈 통행량 df 생성
            returns:
                near_link_spd_count_df: 링크별 인접링크 주행속도 구간 통행량, 평균, 표준편차의 속성정보가 담긴 df
        """
        window_spec = Window.partitionBy('TripId').orderBy('inTime')
        #self.df = self.df.filter((col('inHour') == timecoderange - 1) & (col('outHour') == timecoderange - 1))

        self.df = self.df.filter(col('inHour') == timecoderange - 1)
        n_df = self.df.groupBy('vLinkId').agg(countDistinct('TripId').alias('total_count_per_link'), first('inHour').alias('timecode'), 
                                              first('weekendCode').alias('weekendCode'))
        
        time_link_speed_merge_df = self.df.withColumn('prev_speed', lag('gisSpeed').over(window_spec))
        time_link_speed_merge_df = time_link_speed_merge_df.withColumn('spd_diff', when(col('prev_speed').isNotNull(), col('gisSpeed') - col('prev_speed')))
        time_link_speed_merge_df = time_link_speed_merge_df.join(n_df, on='vLinkId', how='left')
        near_link_5spd_df = time_link_speed_merge_df.filter(abs(col('spd_diff')) < 5.0)
        near_link_5spd_df = near_link_5spd_df.groupBy('vLinkId').agg(countDistinct('TripId').alias('5km_spd_count'))
        near_link_10spd_df = time_link_speed_merge_df.filter((abs(col('spd_diff')) >= 5.0) & (abs(col('spd_diff')) < 10.0))
        near_link_10spd_df = near_link_10spd_df.groupBy('vLinkId').agg(countDistinct('TripId').alias('10km_spd_count'))
        near_link_15spd_df = time_link_speed_merge_df.filter((abs(col('spd_diff')) >= 10.0) & (abs(col('spd_diff')) < 15.0))
        near_link_15spd_df = near_link_15spd_df.groupBy('vLinkId').agg(countDistinct('TripId').alias('15km_spd_count'))
        near_link_20spd_df = time_link_speed_merge_df.filter((abs(col('spd_diff')) >= 15.0) & (abs(col('spd_diff')) < 20.0))
        near_link_20spd_df = near_link_20spd_df.groupBy('vLinkId').agg(countDistinct('TripId').alias('20km_spd_count'))
        near_link_25spd_df = time_link_speed_merge_df.filter((abs(col('spd_diff')) >= 20.0) & (abs(col('spd_diff')) < 25.0))
        near_link_25spd_df = near_link_25spd_df.groupBy('vLinkId').agg(countDistinct('TripId').alias('25km_spd_count'))
        near_link_30spd_df = time_link_speed_merge_df.filter((abs(col('spd_diff')) >= 25.0) & (abs(col('spd_diff')) < 30.0))
        near_link_30spd_df = near_link_30spd_df.groupBy('vLinkId').agg(countDistinct('TripId').alias('30km_spd_count'))
        near_link_35spd_df = time_link_speed_merge_df.filter((abs(col('spd_diff')) >= 30.0) & (abs(col('spd_diff')) < 35.0))
        near_link_35spd_df = near_link_35spd_df.groupBy('vLinkId').agg(countDistinct('TripId').alias('35km_spd_count'))
        near_link_40spd_df = time_link_speed_merge_df.filter((abs(col('spd_diff')) >= 35.0) & (abs(col('spd_diff')) < 40.0))
        near_link_40spd_df = near_link_40spd_df.groupBy('vLinkId').agg(countDistinct('TripId').alias('40km_spd_count'))
        near_link_45spd_df = time_link_speed_merge_df.filter((abs(col('spd_diff')) >= 40.0) & (abs(col('spd_diff')) < 45.0))
        near_link_45spd_df = near_link_45spd_df.groupBy('vLinkId').agg(countDistinct('TripId').alias('45km_spd_count'))
        near_link_50spd_df = time_link_speed_merge_df.filter((abs(col('spd_diff')) >= 45.0) & (abs(col('spd_diff')) < 50.0))
        near_link_50spd_df = near_link_50spd_df.groupBy('vLinkId').agg(countDistinct('TripId').alias('50km_spd_count'))
        near_link_over50spd_df = time_link_speed_merge_df.filter(abs(col('spd_diff')) >= 50.0)
        near_link_over50spd_df = near_link_over50spd_df.groupBy('vLinkId').agg(countDistinct('TripId').alias('ovr50km_spd_count'))

        common_col = 'vLinkId'
        near_link_spd_count_df = near_link_5spd_df.join(near_link_10spd_df, on=common_col, how='outer').join(near_link_15spd_df, on=common_col, how='outer')\
            .join(near_link_20spd_df, on=common_col, how='outer').join(near_link_25spd_df, on=common_col, how='outer')\
            .join(near_link_30spd_df, on=common_col, how='outer').join(near_link_35spd_df, on=common_col, how='outer')\
            .join(near_link_40spd_df, on=common_col, how='outer').join(near_link_45spd_df, on=common_col, how='outer')\
            .join(near_link_50spd_df, on=common_col, how='outer').join(near_link_over50spd_df, on=common_col, how='outer')
        
        near_link_spd_count_df = near_link_spd_count_df.join(n_df, on=common_col, how='inner')
        return near_link_spd_count_df
    
    def near_link_spd(self, timecoderange):
        """
        1차 가공 DTG df로부터 인접 링크 간 주행속도 차이 df 생성
            returns:
                near_link_spd_df: 링크별 인접링크 주행속도 총합, 평균, 표준편차의 속성정보가 담긴 df
        """

        window_spec = Window.partitionBy('TripId').orderBy('inTime')
        # self.df = self.df.filter((col('inHour') == timecoderange - 1) & (col('outHour') == timecoderange - 1)) # intime ~ outtime 인 경우 filter
        self.df = self.df.filter(col('inHour') == timecoderange - 1)

        # 다음 포인트에 대한 속도 값과 비교함.
        time_link_speed_merge_df = self.df.withColumn('next_speed', lead('gisSpeed').over(window_spec))
        time_link_speed_merge_df = time_link_speed_merge_df.withColumn('spd_diff', when(col('next_speed').isNotNull(), col('next_speed') - col('gisSpeed')))
        # 속도 편차 절댓값으로 변환 후 덮어쓰기
        # time_link_speed_merge_df는 트립단위의 spd diff 값 산출을 위한 것
        time_link_speed_merge_df = time_link_speed_merge_df.withColumn('spd_diff', F.abs(col('spd_diff')))
       
       # 링크당 각 트립의 속도 표준 편차 평균 사용
        mean_spd_df = time_link_speed_merge_df.groupBy('vLinkId').agg(
            F.avg('spd_diff').alias('mean_near_link_spd'), first('inHour').alias('timecode')
        )
        near_link_spd_df = time_link_speed_merge_df.join(mean_spd_df, on="vLinkId", how="left").drop('TripID', 'gisSpeed', 'next_speed'
                                                                                                     ,'spd_diff', 'inHour', 'outHour', 'inTime', 'outTime')\
                                                     .withColumnRenamed("mean_near_link_spd", "near_link_spd")

        return near_link_spd_df


def main(args, spark, logger):
    """
    각종 지표를 취합하여 설계된 데이터 테이블을 산출하고 이를 hdfs으로 저장하는 메인 함수
        args:
            args: 사용자 정의 인자 값
            spark: 스파크 세션
            logger: logger 객체
    """
    mode = args.mode.upper()
    net_level = args.net_level
    indicator_metric = SpeedSafetyMetrics(spark, args)
    date_range = utils.get_date_range(args.start_ymd, args.end_ymd)

    if args.indicator == 'link_std_spd':
        result_schema = StructType([
         StructField('vLinkId', StringType(), True),
         StructField('TripID', StringType(), True),
         StructField('inTime', FloatType(), True),
         StructField('outTime', FloatType(), True),
         StructField('gisSpeed', FloatType(), True),
         StructField('inHour', IntegerType(), True),
         StructField('outHour', IntegerType(), True),
         StructField('weekendCode', IntegerType(), True),
         StructField('total_traffic', IntegerType(), True),
         StructField('x_sum_per_link', FloatType(), True)])
        
    elif args.indicator == 'near_link_spd_count':
        result_schema = StructType([
         StructField('vLinkId', StringType(), True),
         StructField('5km_spd_count', IntegerType(), True),
         StructField('10km_spd_count', IntegerType(), True),
         StructField('15km_spd_count', IntegerType(), True),
         StructField('20km_spd_count', IntegerType(), True),
         StructField('25km_spd_count', IntegerType(), True),
         StructField('30km_spd_count', IntegerType(), True),
         StructField('35km_spd_count', IntegerType(), True),
         StructField('40km_spd_count', IntegerType(), True),
         StructField('45km_spd_count', IntegerType(), True),
         StructField('50km_spd_count', IntegerType(), True),
         StructField('ovr50km_spd_count', IntegerType(), True),
         StructField('total_traffic', IntegerType(), True),
         StructField('timecode', FloatType(), True),
         StructField('weekendcode', IntegerType(), True)])
    
    elif args.indicator == 'near_link_spd':
        result_schema = StructType([
         StructField('vLinkId', StringType(), True),
         StructField('weekendcode', IntegerType(), True),
         StructField('near_link_spd', FloatType(), True),
         StructField('timecode', IntegerType(), True)
        ])
        
    else:
        raise ValueError('wrong indicator')
    
    result_df = spark.createDataFrame([], result_schema)
    for idx, ymd in enumerate(date_range):
        logger.info(('Progress: {}/{}').format(idx + 1, len(date_range)))
        y, m, d = ymd.strftime('%Y_%m_%d').split('_')
        if net_level in (6, 8):
            input_path = ('hdfs:/data/DTG/P1_{}_LEV{}/{}/{}/{}/*').format(mode, net_level, y, m, d)
            print('P1_{}_LEV{} processing').format(mode, net_level)

        else:
            print('P1_{}_MOCT_NEW processing').format(mode)
            input_path = ('hdfs:/data/DTG/P1_{}_MOCT_NEW/{}/{}/{}/*').format(mode, y, m, d)

        indicator_metric.p1_load_rdd(input_path)
        indicator_metric.make_final_df()


        if args.indicator == 'link_std_spd':
            indicator_df = indicator_metric.link_spd_std(args.timecode)
        elif args.indicator == 'near_link_spd_count':
            indicator_df = indicator_metric.near_link_spd_count(args.timecode)
        elif args.indicator == 'near_link_spd':
            indicator_df = indicator_metric.near_link_spd(args.timecode)
        
        result_df = result_df.union(indicator_df)
        result_df.count()

    if args.indicator == 'link_std_spd':
        result_df = result_df.withColumn('x_bar', col('x_sum_per_link') / col('total_traffic'))
        result_df = result_df.withColumn('diff', col('gisSpeed') - col('x_bar'))
        result_df = result_df.withColumn('squared_diff', pow(col('diff'), 2))
        result_df = result_df.groupBy('vLinkId').agg(sum('squared_diff').alias('squared_diff_sum'), first('total_traffic').alias('total_traffic'), 
                                                     first('x_bar').alias('avg_spd'), first('inHour').alias('timecode'), first('weekendCode').alias('weekendCode'))
        result_df = result_df.withColumn('std_spd', when(col('squared_diff_sum').isNull() | (col('total_traffic') <= 1), 0)\
                                         .otherwise(sqrt(col('squared_diff_sum') / (col('total_traffic') - 1.0))))
        result_df = result_df.drop('squared_diff', 'diff', 'squared_diff_sum')

    elif args.indicator == 'near_link_spd_count':
        result_df = result_df.groupBy('vLinkId').agg(sum('5km_spd_count').alias('5km_spd_count'), sum('10km_spd_count').alias('10km_spd_count'),
                                                    sum('15km_spd_count').alias('15km_spd_count'), sum('20km_spd_count').alias('20km_spd_count'), 
                                                    sum('25km_spd_count').alias('25km_spd_count'), sum('30km_spd_count').alias('30km_spd_count'), 
                                                    sum('35km_spd_count').alias('35km_spd_count'), sum('40km_spd_count').alias('40km_spd_count'), 
                                                    sum('45km_spd_count').alias('45km_spd_count'), sum('50km_spd_count').alias('50km_spd_count'), 
                                                    sum('ovr50km_spd_count').alias('ovr50km_spd_count'), sum('total_traffic').alias('total_traffic'), 
                                                    first('timecode').alias('timecode'), first('weekendcode').alias('weekendcode'))
        
    elif args.indicator == 'near_link_spd':
        # 링크 당 속도 표준편차의 평균
        result_df = result_df.groupBy('vLinkId').agg(F.avg('near_link_spd').alias('near_link_spd')
                                                     ,first('timecode').alias('timecode')
                                                     ,first('weekendcode').alias('weekendcode')
                                                    )
    
        # vLinkId | timecode | weekendcode | near_link_spd
        result_df = result_df.select('vLinkid', 'timecode', 'weekendcode', 'near_link_spd')

    else:
        raise Exception('Error: choose wrong indicator.')
    
    result_df = result_df.withColumn('timecode', col('timecode').cast(IntegerType()) + 1)
    result_df.show(5)

    logger.info(('{}_{} indicator saved!').format(mode, args.indicator))
    output_file = ('/{}/{}/timecode_{}').format(mode, args.indicator, args.timecode)
    output_path = cfg.OUTPUT_DIR + output_file
    output_rdd = result_df.rdd.map((lambda row: (',').join([str(item) for item in row])))
    output_rdd.coalesce(10).saveAsTextFile(output_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info('start spark session')
    args = utils.get_args()
    APPLICATION_NAME = cfg.APPLICATION_NAME[1] + ('_{}_{}(time range: {}~{}, timecode: {})').format(args.mode, args.indicator, args.start_ymd, args.end_ymd, args.timecode)
    spark = SparkSession.builder.appName(APPLICATION_NAME).config('spark.executor.instances', '10').config('spark.executor.memory', '15g').config('spark.executor.cores', '5').getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    spark.sparkContext.addPyFile('/home/hadoop/odyssey/san/utils.py')
    main(args, spark, logger)
    spark.stop()
