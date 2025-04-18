import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import logging
import math
import numpy as np
import warnings

from utils import get_weekdays_in_same_week
from core import check_valid_dtg, MatchingInvalidError
from accident_matching import AccidentMatching, AccidentDataPreprocessing

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
warnings.simplefilter(action='ignore', category=FutureWarning)


class AccidentMapMatchingProcessor(AccidentDataPreprocessing, AccidentMatching):
    def __init__(self, taas_data_path: str, ps_data_path: str, moct_network_path: str):
        AccidentDataPreprocessing.__init__(self, taas_data_path, ps_data_path, moct_network_path)
        AccidentMatching.__init__(self, radius=100)
        self._ignore_error = True

    @property
    def error_ignore(self):
        return self._ignore_error

    @error_ignore.setter
    def error_ignore(self, value: bool):
        self._ignore_error = value

    def get_data_size(self, chunk=1000000):
        with open(self.ps_data_path, 'r', encoding='utf-8') as f:
            return math.ceil(sum(1 for _ in f) / chunk)

    def accident_day_dataloader(self):
        '''
        3개의 data(taas, ps, moct)를 accidentdatapreprocessing 클래스로 전처리하여 지역, 시간 필터링 df를 산출하는 함수
        returns:
            self.accident_day_df: 사고 당일의 사고 지점 근처의 후보 링크 + 시간대에 대한 궤적 데이터
        '''
        total_chunk_size = self.get_data_size()
        self.get_candidate_links(self.taas_sample_gdf, self.moct_link_gdf)
        self.accident_day_df = gpd.GeoDataFrame(
            columns=['link_id', 'trip_id', 'pointTime', 'speed', 'geometry', 'time_bin_index'],
            geometry='geometry',
            crs="EPSG:5179"
        )

        for ps_near_chunk in tqdm(self.link_ps_merge_sampling(),
                                  total=total_chunk_size,
                                  desc="사고 당일 데이터(gps point) 로드 중..",
                                  unit="per chunk_size"):
            ps_on_candidate_links_with_timebin_gdf = self.candidate_links_with_timebin(
                new_timestamp=self.taas_sample_gdf['unixtime'].values,
                link_ps_merge_near_acctime_gdf=ps_near_chunk
            )
            self.accident_day_df = pd.concat([self.accident_day_df, ps_on_candidate_links_with_timebin_gdf], ignore_index=True)

        if self.error_ignore:
            try:
                check_valid_dtg(candidate_links=self.candidate_links, input_df=self.accident_day_df)
            except MatchingInvalidError:
                print('**[WARNING]** DTG 데이터가 부족하여 정확한 점수를 산출할 수 없습니다.')
        else:
            check_valid_dtg(candidate_links=self.candidate_links, input_df=self.accident_day_df)

        logging.info(f"사고 당일 데이터 로드 완료. 결과: 총 {len(self.accident_day_df)}개 데이터 포인트 추출")

    def get_other_days_score(self, save_score=False):
        '''
        3개의 data(taas, ps, moct)를 accidentdatapreprocessing 클래스로 전처리하여 지역, 시간 필터링 df를 산출하는 함수

        returns:
            self.other_day_df: 사고 당일을 제외한 평일 or 주말의 후보 링크 + 시간대에 대한 궤적 데이터
        '''
        is_first = True
        if is_first:
            input_date_str = self.ps_data_path.split('/')[1].split('_')[1][:-4]  # 사고 날짜 ps_data_path
        else:
            input_date_str = self.ps_data_path.split('\\')[2].split('_')[1][:-4]  # os window 고려

        daystr_ofweeks = get_weekdays_in_same_week(input_date_str)  # datetime 반환
        for idx_day, date in enumerate(daystr_ofweeks):
            # 하루치 df 초기화
            self.other_day_df = gpd.GeoDataFrame(columns=['link_id', 'trip_id', 'pointTime', 'speed', 'geometry', 'time_bin_index'],
                                                 geometry='geometry', crs="EPSG:5179")
            new_date_str = date.strftime("%Y%m%d")
            self.ps_data_path = f'traj_sample/alltraj_{new_date_str}.txt'
            total_chunk_size = self.get_data_size()
            for other_ps_chunk in tqdm(self.remain_days_traj(new_date_str, preprocessing_on_local=True),
                                       total=total_chunk_size,
                                       desc=f"사고 이외 {idx_day + 1}번째 날 데이터(gps point) 로드 중..",
                                       unit="per chunk size"):
                ps_on_candidate_links_with_timebin_gdf = self.candidate_links_with_timebin(
                    new_timestamp=self.new_timestamp,
                    link_ps_merge_near_acctime_gdf=other_ps_chunk
                )
                self.other_day_df = pd.concat([self.other_day_df, ps_on_candidate_links_with_timebin_gdf], ignore_index=True)

            if self.error_ignore:
                try:
                    check_valid_dtg(candidate_links=self.candidate_links, input_df=self.other_day_df)
                except MatchingInvalidError:
                    print('**[WARNING]** DTG 데이터가 부족하여 정확한 점수를 산출할 수 없습니다.')
            else:
                check_valid_dtg(candidate_links=self.candidate_links, input_df=self.accident_day_df)

            other_day_score = self.candidate_link_score(ps_on_candidate_links_gdf=self.other_day_df)
            if save_score:
                other_day_score.to_csv(f'score/other_day_score{idx_day + 1}.csv', index=False)

            logging.info(f"사고 이외 날짜 기반 {idx_day + 1}번째 날 소통 점수 산출 완료")
            yield other_day_score

    def get_accident_day_score_diff(self, save_score=False):
        self.accident_day_dataloader()
        # accident_day_df = self.accident_day_df # 'link_id', 'trip_id', 'pointTime', 'speed', 'geometry', 'time_bin_index'
        logging.info("사고 당일 데이터 기반 소통 점수 산출 중... ")
        accident_day_score = self.candidate_link_score(ps_on_candidate_links_gdf=self.accident_day_df)

        if save_score:  # 점수차이 계산 전 점수까지의 내용 저장
            accident_day_score.to_csv('score/accident_day_score.csv', encoding='cp949', index=False)

        accident_day_score = accident_day_score[['link_id', 'time_bin_index', 'score']]
        accident_day_score_pivot = accident_day_score.pivot(index='link_id', columns='time_bin_index', values='score')
        # 다음 10분 점수와의 차이 계산
        accident_day_score_diff = pd.DataFrame(data=accident_day_score_pivot.values[:, :-1] - accident_day_score_pivot.values[:, 1:],
                                               index=accident_day_score_pivot.index,
                                               columns=accident_day_score_pivot.columns[:-1])
        accident_day_score_diff.columns = accident_day_score_diff.columns.map(lambda x: f"diff_{x}to{x + 1}")

        logging.info("사고 당일 데이터 기반 소통 점수 산출 완료")
        return accident_day_score_diff

    def get_other_day_score_diff_stats(self, accident_day_score, save_score=False):
        '''
        사고일 이외의 날짜들에 대해 시간별 diff 점수의 평균과 표준편차를 계산
        Returns:
            mean_df: diff 점수 평균
            std_df: diff 점수 표준편차
        '''
        scale = 100.0
        day_count = 0

        accident_day_score = accident_day_score * scale
        diff_list = []
        diff_list.append(accident_day_score)

        for other_day_score_per_oneday in self.get_other_days_score(save_score=save_score):
            day_count += 1
            other_day_score_per_oneday = other_day_score_per_oneday[['link_id', 'time_bin_index', 'score']]
            pivot_df = other_day_score_per_oneday.pivot(index='link_id', columns='time_bin_index', values='score')

            # 10분 단위 diff 계산
            score_diff = pd.DataFrame(
                data=pivot_df.values[:, :-1] - pivot_df.values[:, 1:],
                index=pivot_df.index,
                columns=pivot_df.columns[:-1]
            ).fillna(0)
            score_diff.columns = score_diff.columns.map(lambda x: f"diff_{x}to{x + 1}")
            score_diff = score_diff * scale
            diff_list.append(score_diff)

        # 모든 날짜별 diff 합치기
        all_diff_concat = pd.concat(diff_list, axis=0)

        # 평균과 표준편차 계산
        mean_df = all_diff_concat.groupby(all_diff_concat.index).mean()
        std_df = all_diff_concat.groupby(all_diff_concat.index).std()

        logging.info("사고 이외 날짜 소통 점수 평균 및 표준편차 산출 완료")

        # 표준편차 컬럼 이름 변경
        std_df.columns = [f"{col}_std" for col in std_df.columns]
        return mean_df, std_df

    def get_results_score(self, accident_day_score_diff, other_day_score_mean, other_day_score_std=None, method='basic'):
        """
        사고일과 다른 날의 점수 차이를 계산하는 함수
        Args:
            accident_day_score_diff (pd.DataFrame): 사고일 점수 차이
            other_day_score_mean (pd.DataFrame): 다른 날 평균 점수 차이
            other_day_score_std (pd.DataFrame, optional): 다른 날 표준편차. method = zscore 방식일 때 필요
            method (str): 'basic' or 'zscore'
        Returns:
            pd.DataFrame: 차이 점수 결과
        """
        scale = 100.0
        accident_day_score_diff = accident_day_score_diff * scale
        filtered_diff = accident_day_score_diff.join(other_day_score_mean, lsuffix='_a', rsuffix='_o', how='outer')
        a_cols = [col for col in filtered_diff.columns if col.endswith('_a')]
        o_cols = [col for col in filtered_diff.columns if col.endswith('_o')]

        if method == 'zscore':
            if other_day_score_std is None:
                raise ValueError("Z-score 방식을 사용하려면 표준편차 데이터가 필요합니다. ")
            filtered_diff = filtered_diff.join(other_day_score_std, rsuffix='_std')
            std_cols = [col for col in filtered_diff.columns if col.endswith('_std')]
            common_cols = [col.replace('_a', '') for col in a_cols if f"{col.replace('_a', '_o')}" in o_cols and f"{col.replace('_a', '_std')}" in std_cols]
        elif method == 'basic':
            common_cols = [col.replace('_a', '') for col in a_cols if f"{col.replace('_a', '_o')}" in o_cols]

        diff_result = pd.DataFrame(index=filtered_diff.index)
        for col in common_cols:
            col_a = f"{col}_a"
            col_o = f"{col}_o"
            if method == 'zscore':  # zscore 기준 점수 산출
                col_std = f"{col}_std"
                diff_result[col] = (filtered_diff[col_a] - filtered_diff[col_o]) / filtered_diff[col_std]
            else:  # 변화량 기준 점수 산출
                diff_result[col] = (filtered_diff[col_a] - filtered_diff[col_o]) / filtered_diff[col_a]

        diff_result.replace([np.inf, -np.inf], np.nan, inplace=True)
        diff_result.to_csv('result/delta_score.csv', encoding='cp949')
        return diff_result

    def get_confidence_from_zscore(self, zscore_df: pd.DataFrame):
        """
        confidence score 생성
        Returns:
            confidence_df: [0, 1] 범위의 confidence 점수 (양수 Z-score인 부분만, 나머지는 0)
        """
        df = zscore_df.copy()
        df = df.reset_index()
        # 점수 [0, 1]로 변환 - minmax 정규화
        min_val = df['result_score'].min()
        max_val = df['result_score'].max()
        df['normalized_score'] = (df['result_score'] - min_val) / (max_val - min_val + 1e-8)  # 0~1

        # softmax 적용
        temp = 0.5
        e = np.exp(df['normalized_score'] / temp)
        df['confidence'] = e / e.sum()

        return df[['link_id', 'confidence']]

    def mapmatching_result(self, from_saved=False):
        if from_saved:
            # 사고 당일 점수 차이 계산
            logging.info("저장된 소통 점수로부터 사고-링크 구간 매칭 프로세스를 시작합니다.")
            accident_day_score = pd.read_csv("score/accident_day_score.csv")
            accident_day_score = accident_day_score[['link_id', 'time_bin_index', 'score']]
            accident_day_score_pivot = accident_day_score.pivot(index='link_id', columns='time_bin_index', values='score')

            accident_day_score_diff = pd.DataFrame(
                data=accident_day_score_pivot.values[:, :-1] - accident_day_score_pivot.values[:, 1:],
                index=accident_day_score_pivot.index,
                columns=accident_day_score_pivot.columns[:-1]
            )
            accident_day_score_diff.columns = accident_day_score_diff.columns.map(lambda x: f"diff_{x}to{x + 1}")

            day_count = 0
            is_first = True
            for i in range(1, 5):
                other_day_score = pd.read_csv(f"score/other_day_score{i}.csv")
                day_count += 1
                other_day_score = other_day_score[['link_id', 'time_bin_index', 'score']]
                other_day_score_pivot = other_day_score.pivot(index='link_id', columns='time_bin_index', values='score')

                # 10분 단위 점수 차이 계산
                other_day_score_diff = pd.DataFrame(
                    data=other_day_score_pivot.values[:, :-1] - other_day_score_pivot.values[:, 1:],
                    index=other_day_score_pivot.index,
                    columns=other_day_score_pivot.columns[:-1]
                ).fillna(0)
                other_day_score_diff.columns = other_day_score_diff.columns.map(lambda x: f"diff_{x}to{x + 1}")
                if is_first:
                    other_day_score_mean = other_day_score_diff
                    is_first = False
                else:
                    other_day_score_mean = other_day_score_mean + other_day_score_diff  # 값 누적 합
            # 평균 계산
            other_day_score_mean /= day_count
            other_day_score_mean = other_day_score_mean.replace(0, np.nan)

            result = self.get_results_score(aaccident_day_score_diff=accident_day_score_diff, other_day_score_mean=other_day_score_mean, method='zscore')

        else:
            # 둘 다 pivot 테이블
            accident_day_score_diff = self.get_accident_day_score_diff(save_score=True)
            other_day_score_mean, other_day_score_std = self.get_other_day_score_diff_stats(accident_day_score=accident_day_score_diff, save_score=True)
            final_score = self.get_results_score(accident_day_score_diff, other_day_score_mean, other_day_score_std, method='zscore')

            # 사고 예상 시점 (diff_0to1 ~ diff_8to9)
            interest_cols = [col for col in final_score.columns if 'diff_' in col and 'to' in col and int(col.split('_')[1].split('to')[0]) >= 0]
            # 사고 예상 timebin 내의 z-score 최대값
            max_score = final_score[interest_cols].max(axis=1, skipna=True)
            result = pd.DataFrame({
                "result_score": max_score
            })

        return result

    def run(self):
        # link: zscore 형태의 df
        result = self.mapmatching_result(from_saved=False)
        result.to_csv('result/final_score.csv', encoding='cp949')
        confidence = self.get_confidence_from_zscore(result)
        confidence.to_csv('result/confidence_score.csv', encoding='cp949')

        result = result.reset_index()
        final_df = pd.merge(result, confidence, on='link_id', how='inner')
        final_df = final_df.sort_values(by="result_score", ascending=False)
        final_df.to_csv('result/results.csv', encoding='cp949')

        print('<사고-링크 구간 매칭 예측 링크>')
        print(f"Top 1 예측 링크 id: {final_df.iloc[0]['link_id']}  confidence: {final_df.iloc[0]['confidence']:.2f}")


def main():
    taas_data_path = 'taas_dataset/20231212.csv'
    ps_data_path = 'traj_sample/alltraj_20231212.txt'
    moct_network_path = 'moct_link/link'

    accidentlinkmatching = AccidentMapMatchingProcessor(taas_data_path, ps_data_path, moct_network_path)
    accidentlinkmatching.error_ignore = True
    accidentlinkmatching.run()


if __name__ == '__main__':
    main()
