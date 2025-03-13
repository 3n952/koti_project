#!/bin/bash

while getopts "m:i:" opt; do
  case $opt in
    m) mode="$OPTARG" ;;
    i) indicator="$OPTARG" ;;
    \?) echo "사용법: $0 -m <mode> -i <indicator> -t <[9~17]> "; exit 1 ;;
  esac
done

echo "mode: $mode"
echo "indicator: $indicator"
echo "timecode range: [9,17]"

for i in {9..17}
do
  echo "Running: hdfs dfs -text hdfs:/directory /home/hadoop/directory../timecode_$i/*"
  hdfs dfs -text /odyssey/san/metric/$mode/$indicator/timecode_$i/* > /home/hadoop/odyssey/san/metric_samples/${mode}_${indicator}_$i.txt
done