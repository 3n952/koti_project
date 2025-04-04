import pandas as pd
from process_run import AccidentMapMatchingProcessor
from tqdm import tqdm
import logging
import geopandas as gpd

def recall_top5(ground_truth: str, prediction_df: pd.DataFrame) -> float:
    """
    prediction_df에서 점수가 높은 순으로 Top 5를 추출하고,
    정답 링크아이디가 포함되어 있는지로 Recall@5 계산

    Parameters:
    - ground_truth: 실제 정답 링크아이디 (문자열)
    - prediction_df: 'link_id' (str), 'score' (float) 컬럼을 가진 DataFrame

    Returns:
    - Recall@5 값 (0.0 또는 1.0)
    """
    top5 = prediction_df.sort_values(by='result_score', ascending=False).head(5)
    top5_ids = top5['link_id'].tolist()
    return 1.0 if ground_truth in top5_ids else 0.0

def recall_top10(ground_truth: str, prediction_df: pd.DataFrame) -> float:
    """
    prediction_df에서 점수가 높은 순으로 Top 5를 추출하고,
    정답 링크아이디가 포함되어 있는지로 Recall@5 계산

    Parameters:
    - ground_truth: 실제 정답 링크아이디 (문자열)
    - prediction_df: 'link_id' (str), 'score' (float) 컬럼을 가진 DataFrame

    Returns:
    - Recall@5 값 (0.0 또는 1.0)
    """
    top10 = prediction_df.sort_values(by='result_score', ascending=False).head(10)
    top10_ids = top10['link_id'].tolist()
    return 1.0 if ground_truth in top10_ids else 0.0

def mrr_top10(ground_truth: str, prediction_df: pd.DataFrame) -> float:
    """
    prediction_df에서 점수가 높은 순으로 Top 10을 추출하고,
    정답의 순위를 바탕으로 MRR@10 계산

    Parameters:
    - ground_truth: 실제 정답 링크아이디 (문자열)
    - prediction_df: 'link_id' (str), 'score' (float) 컬럼을 가진 DataFrame

    Returns:
    - MRR@10 값 (0.0 ~ 1.0)
    """
    top10 = prediction_df.sort_values(by='result_score', ascending=False).head(10)
    top10_ids = top10['link_id'].tolist()
    
    if ground_truth in top10_ids:
        rank = top10_ids.index(ground_truth) + 1  # 인덱스는 0부터 시작하니까 +1
        return 1.0 / rank
    else:
        return 0.0

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    tass_data_path = 'tass_dataset/taas_under_100.csv'
    ps_data_path = 'traj_sample/alltraj_20231211.txt'
    moct_network_path = 'moct_link/link'
    taas_sample = pd.read_csv(tass_data_path, encoding='cp949')
    final_result = []
    total_volume = float(len(taas_sample))
    answer_count = 0.0

    accidentlinkmatching = AccidentMapMatchingProcessor(tass_data_path, ps_data_path, moct_network_path)

    logging.info(f'총 {total_volume}개 테스트 샘플에 대한 평가지표 산출중')
    for idx, (_, row) in enumerate(taas_sample.iterrows()):
        answer_link_id = str(row['link_id'])
        row_df = pd.DataFrame([row])
        tass_df = row_df[['acdnt_dd_dc', 'occrrnc_time_dc', 'x_crdnt_crdnt', 'y_crdnt_crdnt']]
        tass_df.loc[:, 'occrrnc_time_dc'] = tass_df['occrrnc_time_dc'].str.replace('시', '', regex=True) # timecode
        tass_df.loc[:, 'datetime'] = pd.to_datetime(tass_df['acdnt_dd_dc'] + ' ' + 
                                                            tass_df['occrrnc_time_dc'] + ':00:00') # unixtime 변환
        tass_df.loc[:, 'unixtime']  = tass_df['datetime'].astype('int64') // 10**9 - (9 * 3600) # KST 기준 맞추기
        accidentlinkmatching.tass_sample_gdf = gpd.GeoDataFrame(
            tass_df,
            geometry=gpd.points_from_xy(row_df['x_crdnt_crdnt'], row_df['y_crdnt_crdnt']),
            crs="EPSG:5179"
        )
        tass_sample_timestamp = accidentlinkmatching.tass_sample_gdf['unixtime'].iloc[0]
        tass_sample_timecode = accidentlinkmatching.tass_sample_gdf['occrrnc_time_dc'].astype(int).iloc[0]
        accidentlinkmatching.uptime = tass_sample_timestamp + 5400
        accidentlinkmatching.downtime = tass_sample_timestamp - 1800
        accidentlinkmatching.timecode = tass_sample_timecode

        result = accidentlinkmatching.mapmatching_result(from_saved=False)
        result = result.reset_index()
        answer_count += recall_top5(answer_link_id, result) # float
        metric_ = answer_count / (idx+1)
        print(f'{idx+1}번째 샘플까지의 recall@5: {metric_}')

    metric = answer_count / total_volume
    logging.info(f"총 {total_volume}개의 테스트 샘플에 대한 평가지표 산출 완료.\nrecall@5: {metric}")

    
