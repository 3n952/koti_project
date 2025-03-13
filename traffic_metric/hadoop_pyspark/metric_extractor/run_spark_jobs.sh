#!/bin/bash
# 우진과장님 방식으로 link_std_spd 구하는 경우 link_std_spd_new.py 스크립트 실행하기

while getopts "m:i:s:e:" opt; do
  case $opt in
    m) mode="$OPTARG" ;;
    i) indicator="$OPTARG" ;;
    s) start_ymd="$OPTARG" ;;
    e) end_ymd="$OPTARG" ;;
    \?) echo "주의: 해당 스크립트는 speed_safety_metric을 위한 스크립트입니다. 사용법: $0 -m <mode> -i <indicator> -s <start_ymd> -e <end_ymd> "; exit 1 ;;
  esac
done

echo "mode: $mode"
echo "indicator: $indicator"
echo "start_ymd: $start_ymd"
echo "end_ymd: $end_ymd"

for i in {9..17}
do
  echo "timecode: $i"
  echo "Running: sparkjob main.py --mode $mode --indicator $indicator --start_ymd $start_ymd --end_ymd $end_ymd --timecode $i"
  spark-submit --deploy-mode client --master yarn main.py --mode $mode --indicator $indicator --start_ymd $start_ymd --end_ymd $end_ymd --timecode $i
done