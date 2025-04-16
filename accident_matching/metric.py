import pandas as pd
import logging
import geopandas as gpd
import warnings

from process_run import AccidentMapMatchingProcessor

warnings.simplefilter(action='ignore', category=FutureWarning)


def recall_topN(n: int, ground_truth: str, prediction_df: pd.DataFrame) -> float:
    """
    prediction_df에서 점수가 높은 순으로 Top 5를 추출하고,
    정답 링크아이디가 포함되어 있는지로 Recall@5 계산

    Parameters:
    - ground_truth: 실제 정답 링크아이디 (문자열)
    - prediction_df: 'link_id' (str), 'score' (float) 컬럼을 가진 DataFrame

    Returns:
    - Recall@5 값 (0.0 또는 1.0)
    """
    topN = prediction_df.sort_values(by='result_score', ascending=False).head(n)
    topN_ids = topN['link_id'].tolist()
    return 1.0 if ground_truth in topN_ids else 0.0


def mrr_topN(n: int, ground_truth: str, prediction_df: pd.DataFrame) -> float:
    """
    prediction_df에서 점수가 높은 순으로 Top 10을 추출하고,
    정답의 순위를 바탕으로 MRR@N 계산

    Parameters:
    - ground_truth: 실제 정답 링크아이디 (문자열)
    - prediction_df: 'link_id' (str), 'score' (float) 컬럼을 가진 DataFrame

    Returns:
    - MRR@N 값 (0.0 ~ 1.0)
    """
    topN = prediction_df.sort_values(by='result_score', ascending=False).head(n)
    topN_ids = topN['link_id'].tolist()
    if ground_truth in topN_ids:
        rank = topN_ids.index(ground_truth) + 1  # 인덱스는 0부터 시작하니까 +1
        return 1.0 / rank
    else:
        return 0.0


def evaluate(data: pd.DataFrame, processor_instance: AccidentMapMatchingProcessor, total_volume: int):
    '''
    recall@5, recall@1에 대한 평가지표를 산출
    '''
    top5_answer_count = 0.0
    top1_answer_count = 0.0

    for idx, (_, row) in enumerate(data.iterrows()):
        answer_link_id = str(row['link_id'])
        row_df = pd.DataFrame([row])
        taas_df = row_df[['acdnt_dd_dc', 'occrrnc_time_dc', 'x_crdnt_crdnt', 'y_crdnt_crdnt']]
        taas_df.loc[:, 'occrrnc_time_dc'] = taas_df['occrrnc_time_dc'].str.replace('시', '', regex=True)  # timecode
        taas_df.loc[:, 'datetime'] = pd.to_datetime(taas_df['acdnt_dd_dc'] + ' ' + taas_df['occrrnc_time_dc'] + ':00:00')  # unixtime 변환
        taas_df.loc[:, 'unixtime'] = taas_df['datetime'].astype('int64') // 10**9 - (9 * 3600)  # KST 기준 맞추기
        processor_instance.taas_sample_gdf = gpd.GeoDataFrame(
            taas_df,
            geometry=gpd.points_from_xy(row_df['x_crdnt_crdnt'], row_df['y_crdnt_crdnt']),
            crs="EPSG:5179")
        logging.info(f"{idx + 1} 번째 사고 데이터\n{processor_instance.taas_sample_gdf.head()}")
        taas_sample_timestamp = processor_instance.taas_sample_gdf['unixtime'].iloc[0]
        taas_sample_timecode = processor_instance.taas_sample_gdf['occrrnc_time_dc'].astype(int).iloc[0]
        processor_instance.uptime = taas_sample_timestamp + 5400
        processor_instance.downtime = taas_sample_timestamp - 1800
        processor_instance.timecode = taas_sample_timecode

        result = processor_instance.mapmatching_result(from_saved=False)
        result = result.reset_index()
        top5_answer_count += recall_topN(n=5, ground_truth=answer_link_id, prediction_df=result)
        top1_answer_count += recall_topN(n=1, ground_truth=answer_link_id, prediction_df=result)

        metric_top5 = top5_answer_count / (idx + 1)
        metric_top1 = top1_answer_count / (idx + 1)
        logging.info(f'[{idx + 1}번째 샘플까지]\nrecall@5: {metric_top5}\trecall@1: {metric_top1}')

    metric_result_top5 = top5_answer_count / total_volume
    metric_result_top1 = top1_answer_count / total_volume

    return metric_result_top5, metric_result_top1


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    taas_data_path = 'taas_dataset/taas_under_100.csv'
    ps_data_path = 'traj_sample/alltraj_20231211.txt'
    moct_network_path = 'moct_link/link'
    taas_sample = pd.read_csv(taas_data_path, encoding='cp949')
    total_volume = int(len(taas_sample))
    accidentlinkmatching = AccidentMapMatchingProcessor(taas_data_path, ps_data_path, moct_network_path)

    # 지표 계산
    logging.info(f'총 {total_volume}개 테스트 샘플에 대한 평가지표 산출중')
    metric_result_top5, metric_result_top1 = evaluate(data=taas_sample,
                                                      processor_instance=accidentlinkmatching,
                                                      total_volume=total_volume)
    logging.info(f"총 {total_volume}개의 테스트 샘플에 대한 평가지표 산출 완료.\nrecall@5: {metric_result_top5}\trecall@1: {metric_result_top1}")
