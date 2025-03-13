# -*- coding: utf-8 -*-
import config as cfg, utils, logging
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, IntegerType, LongType
from pyspark.sql.functions import col, countDistinct, from_unixtime, hour, udf, sum, when, first, sqrt

## 우진 과장님의 로직과 맞추기 위한 link_std_spd용 파이스파크 스크립트

class SpeedSafetyMetrics2:
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
         StructField('vLinkId', LongType(), True),
         StructField('inTime', FloatType(), True),
         StructField('outTime', FloatType(), True),
         StructField('gisSpeed', FloatType(), True)])
        rdd = self.spark.sparkContext.textFile(rdd_file_path)
        parsed_rdd = rdd.mapPartitions(p1_parse_line)
        self.df = self.spark.read.schema(schema).csv(parsed_rdd)

    def make_final_df(self):
        """
        1차 가공된 dataFrame의 data에서 이상치를 제거하고 목적에 맞게 전처리
        """
        from utils import is_weekend
        is_weekend_udf = udf(is_weekend, IntegerType())
        self.df = self.df.filter((self.df['vLinkId'] != '-1') & (self.df['matchingRate'] >= 90))
        self.df = self.df.drop('matchingRate')
        self.df = self.df.filter(col('gisSpeed').cast('float') != -1.0)
        self.df = self.df.filter(col('inTime').cast('float') < col('outTime').cast('float'))
        self.df = self.df.withColumn('inHour', hour(from_unixtime(col('inTime'), 'yyyy-MM-dd HH:mm:ss')))\
                         .withColumn('outHour', hour(from_unixtime(col('outTime'), 'yyyy-MM-dd HH:mm:ss')))\
                         .withColumn('weekendCode', is_weekend_udf(col('inTime')))
       
    def prev_link_spd_std(self, timecoderange):
        """
        1차 가공 DTG df로부터 링크 별 속도 표준편차 df 생성
            returns:
                link_spd_std_df: 링크별 통행량, 통행속도 총합의 속성정보가 담긴 df
        """
        self.df = self.df.filter((col('inHour') == timecoderange - 1))
        n_df = self.df.groupBy('vLinkId').agg(countDistinct('TripId').alias('n'))
        x_bar_df = self.df.groupBy('vLinkId').agg(sum('gisSpeed').alias('x_sum_per_link'))
        link_spd_std_df = self.df.join(n_df, on='vLinkId', how='left').join(x_bar_df, on='vLinkId', how='left')
        link_spd_std_df = link_spd_std_df.withColumnRenamed('n', 'total_traffic')

        return link_spd_std_df


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
    indicator_metric = SpeedSafetyMetrics2(spark, args)
    date_range = utils.get_date_range(args.start_ymd, args.end_ymd)
    
    if args.indicator == 'prev_link_std_spd':
        result_schema = StructType([
         StructField('vLinkId', LongType(), True),
         StructField('TripID', StringType(), True),
         StructField('inTime', FloatType(), True),
         StructField('outTime', FloatType(), True),
         StructField('gisSpeed', FloatType(), True),
         StructField('inHour', IntegerType(), True),
         StructField('outHour', IntegerType(), True),
         StructField('weekendCode', IntegerType(), True),
         StructField('total_traffic', IntegerType(), True),
         StructField('x_sum_per_link', FloatType(), True)])
        
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
        indicator_df = indicator_metric.prev_link_spd_std(args.timecode)

        result_df = result_df.union(indicator_df)
        result_df.count()

    total_spd_df = result_df.groupBy('vLinkId').agg(sum('x_sum_per_link').alias('total_spd_sum'))
    total_traffic_df = result_df.groupBy('vLinkId').agg(sum('total_traffic').alias('total_traffic_sum'))
    
    result_df = result_df.join(total_spd_df, on='vLinkId', how='left').join(total_traffic_df, on='vLinkId', how='left')

    result_df = result_df.withColumn('x_bar', col('total_spd_sum') / col('total_traffic_sum'))
    result_df = result_df.withColumn('diff', col('gisSpeed') - col('x_bar'))
    result_df = result_df.withColumn('squared_diff', pow(col('diff'), 2))
    result_df = result_df.groupBy('vLinkId').agg(sum('squared_diff').alias('squared_diff_sum'), first('total_traffic_sum').alias('total_traffic'), 
                                                     first('x_bar').alias('avg_spd'), first('inHour').alias('timecode'), first('weekendCode').alias('weekendCode'))
    result_df = result_df.withColumn('std_spd', when(col('squared_diff_sum').isNull() | (col('total_traffic') <= 1), 0)\
                                     .otherwise(sqrt(col('squared_diff_sum') / (col('total_traffic') - 1.0))))
    result_df = result_df.drop('squared_diff', 'diff', 'squared_diff_sum')
    
    result_df = result_df.withColumn('timecode', col('timecode') + 1)
    
    # result_df columns -> vLinkId | total_traffic | avg_spd | timecode | weekendcode | std_spd
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
    APPLICATION_NAME = cfg.APPLICATION_NAME[1] + ('_{}_{}(time range: {}~{}, timecode: {})')\
                        .format(args.mode, args.indicator, args.start_ymd, args.end_ymd, args.timecode)
    
    spark = SparkSession.builder.appName(APPLICATION_NAME).config('spark.executor.instances', '10')\
            .config('spark.executor.memory', '15g')\
            .config('spark.executor.cores', '5').getOrCreate()
    
    spark.sparkContext.setLogLevel('WARN')
    spark.sparkContext.addPyFile('/home/hadoop/odyssey/san/utils.py')
    main(args, spark, logger)
    spark.stop()
