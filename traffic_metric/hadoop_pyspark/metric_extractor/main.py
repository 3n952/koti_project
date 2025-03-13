# -*- coding: utf-8 -*-
import config as cfg
import logging
from metric_modules.eleven_driving_metric import main as eleven_driving_metric_calculator
from metric_modules.speed_safety_metric import main as speed_safety_metric_calculator
from metric_modules.link_std_spd_new import main as link_std_spd_new_calculator #우진 과장님 방식 -> 전체 관심 시간에 대한 모든 값을 통계냄.
from utils import get_args

from pyspark.sql import SparkSession

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.info('start spark session')

    args = get_args()

    if args.indicator in ['link_std_spd', 'near_link_spd', 'near_link_spd_count', 'prev_link_std_spd']:
        APPLICATION_NAME = cfg.APPLICATION_NAME[1]+"_{}_{}(time range: {}~{})".format(args.mode, args.indicator, args.start_ymd, args.end_ymd)
    else:
        APPLICATION_NAME = cfg.APPLICATION_NAME[0]+"_{}_{}(time range: {}~{})".format(args.mode, args.indicator, args.start_ymd, args.end_ymd)

    # session build
    spark = SparkSession.builder.appName(APPLICATION_NAME) \
    .config("spark.executor.instances", "10") \
    .config("spark.executor.memory", "20g") \
    .config("spark.executor.cores", '5') \
    .config("spark.yarn.am.memory", "3g") \
    .getOrCreate()

    # PySpark 로그 레벨 낮추기 (INFO 메시지 제거)
    spark.sparkContext.setLogLevel("WARN")
    # Pyspark UDF 관련 참조 문제 해결 - addpyfile
    spark.sparkContext.addPyFile("/home/hadoop/odyssey/san/utils.py")

    # 지표 산출
    if args.indicator in ['link_std_spd', 'near_link_spd', 'near_link_spd_count']:
        speed_safety_metric_calculator(args, spark, logger)

    # 이전 방식(우진 과장님) 적용 링크 속도 표준편차 산출    
    elif args.indicator in ['prev_link_std_spd']:
        print('woojin preprocessing')
        link_std_spd_new_calculator(args, spark, logger)

    elif args.indicator in ['ovrspd', 'ltrm_ovrspd', 'rapid_start', \
                            'rapid_acc', 'rapid_decel', 'rapid_stop', \
                            'rapid_lane_change', 'rapid_overtake', 'rapid_turn', 'rapid_Uturn']:
        eleven_driving_metric_calculator(args, spark, logger)

    else:
        print('wrong indicator')

    # session exit
    spark.stop()
    