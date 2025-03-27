import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import logging
import math 
import numpy as np
from utils import get_weekdays_in_same_week

from accident_matching import AccidentMatching, AccidentDataPreprocessing

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class AccidentMapMatchingProcessor(AccidentDataPreprocessing, AccidentMatching):
    def __init__(self, tass_data_path: str, ps_data_path: str, moct_network_path: str):
        AccidentDataPreprocessing.__init__(self, tass_data_path, ps_data_path, moct_network_path)
        AccidentMatching.__init__(self, radius=300.0)

    def get_data_size(self, chunk=10000000):
        def count_lines(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
            
        count_line = count_lines(self.ps_data_path)
        total_chunk_size = math.ceil(count_line / chunk)

        return total_chunk_size
    
    def accident_day_dataloader(self):
        '''
        3개의 data(tass, ps, moct)를 accidentdatapreprocessing 클래스로 전처리하여 지역, 시간 필터링 df를 산출하는 함수
        
        returns: 
            self.accident_day_df: 사고 당일의 사고 지점 근처의 후보 링크 + 시간대에 대한 궤적 데이터
        '''

        total_chunk_size = self.get_data_size()
        self.get_candidate_links(self.tass_sample_gdf, self.moct_link_gdf)

        self.accident_day_df = gpd.GeoDataFrame(columns=['link_id', 'trip_id', 'pointTime', 'speed', 'geometry', 'time_bin_index'], 
                             geometry='geometry', crs="EPSG:5179")

        for ps_near_chunk in tqdm(self.link_ps_merge_sampling(), 
                                  total=total_chunk_size, 
                                  desc="사고 당일 데이터(gps point) 로드 중..", 
                                  unit="per chunk_size 10000000"):
            ps_on_candidate_links_with_timebin_gdf = self.candidate_links_with_timebin(
                new_timestamp=self.tass_sample_gdf['unixtime'].values, 
                link_ps_merge_near_acctime_gdf=ps_near_chunk
            )
            self.accident_day_df = pd.concat([self.accident_day_df, ps_on_candidate_links_with_timebin_gdf], ignore_index=True)

        logging.info(f"사고 당일 데이터 로드 완료. 결과: 총 {len(self.accident_day_df)}개 데이터 포인트 추출")

    def get_other_days_score(self, save_score=False):
        '''
        3개의 data(tass, ps, moct)를 accidentdatapreprocessing 클래스로 전처리하여 지역, 시간 필터링 df를 산출하는 함수

        returns:
            self.other_day_df: 사고 당일을 제외한 평일 or 주말의 후보 링크 + 시간대에 대한 궤적 데이터
        '''
        is_first = True
        
        if is_first:
            input_date_str = self.ps_data_path.split('/')[1].split('_')[1][:-4] # 사고 날짜 ps_data_path
        else:
            input_date_str = self.ps_data_path.split('\\')[2].split('_')[1][:-4] # os window 고려

        daystr_ofweeks = get_weekdays_in_same_week(input_date_str) # datetime 반환
        for idx_day, date in enumerate(daystr_ofweeks):
            # 하루치 df 초기화
            self.other_day_df = gpd.GeoDataFrame(columns=['link_id', 'trip_id', 'pointTime', 'speed', 'geometry', 'time_bin_index'], 
                             geometry='geometry', crs="EPSG:5179")
            new_date_str = date.strftime("%Y%m%d")
            self.ps_data_path = f'traj_sample/alltraj_{new_date_str}.txt'
            total_chunk_size = self.get_data_size()
            for other_ps_chunk in tqdm(self.remain_days_traj(new_date_str, preprocessing_on_local=True),
                                   total=total_chunk_size, 
                                   desc=f"사고 이외 {idx_day+1}번째 날 데이터(gps point) 로드 중..", 
                                   unit="per chunk size 10000000"):
                ps_on_candidate_links_with_timebin_gdf = self.candidate_links_with_timebin(
                    new_timestamp=self.new_timestamp,  
                    link_ps_merge_near_acctime_gdf=other_ps_chunk
                )
                self.other_day_df = pd.concat([self.other_day_df, ps_on_candidate_links_with_timebin_gdf], ignore_index=True)

            other_day_score = self.candidate_link_score(ps_on_candidate_links_gdf=self.other_day_df)
            if save_score:
                other_day_score.to_csv(f'score/other_day_score{idx_day+1}.csv', index=False)

            logging.info(f"사고 이외 날짜 기반 {idx_day+1}번째 날 소통 점수 산출 완료")
            yield other_day_score

    def get_accident_day_score_diff(self, save_score=False):
        self.accident_day_dataloader()
        #accident_day_df = self.accident_day_df # 'link_id', 'trip_id', 'pointTime', 'speed', 'geometry', 'time_bin_index'
        logging.info(f"사고 당일 데이터 기반 소통 점수 산출 중... ")
        accident_day_score = self.candidate_link_score(ps_on_candidate_links_gdf=self.accident_day_df)
        
        if save_score: #점수차이 계산 전 점수까지의 내용 저장
            accident_day_score.to_csv(f'score/accident_day_score.csv', encoding='cp949', index=False)

        accident_day_score = accident_day_score[['link_id', 'time_bin_index', 'score']]
        accident_day_score_pivot = accident_day_score.pivot(index='link_id', columns='time_bin_index', values='score')
        # 다음 10분 점수와의 차이 계산 
        accident_day_score_diff = pd.DataFrame(
                data = accident_day_score_pivot.values[:, :-1] - accident_day_score_pivot.values[:, 1:],
                index = accident_day_score_pivot.index,
                columns = accident_day_score_pivot.columns[:-1]
            )
            
        accident_day_score_diff.columns = accident_day_score_diff.columns.map(lambda x: f"diff_{x}to{x+1}")

        logging.info(f"사고 당일 데이터 기반 소통 점수 산출 완료")
        return accident_day_score_diff
    
    def get_other_day_score_diff_mean(self, save_score=False):
        day_count = 0
        is_first = True
        for other_day_score_per_oneday in self.get_other_days_score(save_score=save_score):
            day_count += 1
            other_day_score_per_oneday = other_day_score_per_oneday[['link_id', 'time_bin_index', 'score']]
            other_day_score_per_oneday_pivot = other_day_score_per_oneday.pivot(index='link_id', columns='time_bin_index', values='score')

            # 10분 단위 점수 차이 계산
            other_day_score_diff = pd.DataFrame(
                data = other_day_score_per_oneday_pivot.values[:, :-1] - other_day_score_per_oneday_pivot.values[:, 1:],
                index = other_day_score_per_oneday_pivot.index,
                columns = other_day_score_per_oneday_pivot.columns[:-1]
            ).fillna(0)
            other_day_score_diff.columns = other_day_score_diff.columns.map(lambda x: f"diff_{x}to{x+1}")
            if is_first:
                other_day_score_mean = other_day_score_diff
                is_first = False
            else:
                other_day_score_mean = other_day_score_mean + other_day_score_diff # 값 누적 합

        # 평균 계산
        other_day_score_mean /= day_count
        other_day_score_mean = other_day_score_mean.replace(0, np.nan)
        logging.info(f"사고 이외 날짜 소통 점수 평균 산출 완료")

        return other_day_score_mean

    def mapmatching_result(self, from_saved=True):

        def get_results_score(accident_day_score_diff, other_day_score_mean):
            logging.info(f"사고-링크 구간 링크 매칭 중...")
            # 사고 당일과 나머지 날을 비교
            filtered_accident_day_score_diff = accident_day_score_diff
            filtered_other_day_score_mean = other_day_score_mean
            filtered_diff = filtered_accident_day_score_diff.join(filtered_other_day_score_mean, lsuffix='_a', rsuffix='_o', how='outer')
            a_cols = [col for col in filtered_diff.columns if col.endswith('_a')]
            o_cols = [col for col in filtered_diff.columns if col.endswith('_o')]
            common_cols = [col.replace('_a', '') for col in a_cols if col.replace('_a', '_o') in o_cols]

            # 다른날과 비교하여 변화량이 얼마나 되는 지 점수화 (추후에 z-score로 바꿔야 하는 부분 -> 이걸 또 어떻게 활용해서 최종 link에 붙힐지 고민)
            diff_result = pd.DataFrame(index=filtered_diff.index)
            for col in common_cols:
                col_a = f"{col}_a"
                col_o = f"{col}_o"
                diff_result[col] = (filtered_diff[col_a] - filtered_diff[col_o]) / filtered_diff[col_a]
            diff_result.replace([np.inf, -np.inf], np.nan, inplace=True)
            diff_result.to_csv('result/delta_score.csv', encoding='cp949', index=False)
    
            # 사고 예상 시점 (diff_0to1 ~ diff_8to9)
            positive_cols = [col for col in diff_result.columns if 'diff_' in col and 'to' in col and int(col.split('_')[1].split('to')[0]) >= 0]
            # 사고 예상 이전 시점(diff_-3to-2 ~ diff_-1to0)
            negative_cols = [col for col in diff_result.columns if 'diff_' in col and 'to' in col and int(col.split('_')[1].split('to')[0]) < 0]
            max_pos = diff_result[positive_cols].max(axis=1, skipna=True)
            min_neg = diff_result[negative_cols].min(axis=1, skipna=True)

            diff_between_max_and_min = max_pos - min_neg
            result = pd.DataFrame({
                "result_score": diff_between_max_and_min
            })
            return result
        
        if from_saved:
            # 사고 당일 점수 차이 계산
            logging.info(f"저장된 소통 점수로부터 사고-링크 구간 매칭 프로세스를 시작합니다.")
            accident_day_score = pd.read_csv("score/accident_day_score.csv")
            accident_day_score = accident_day_score[['link_id', 'time_bin_index', 'score']]
            accident_day_score_pivot = accident_day_score.pivot(index='link_id', columns='time_bin_index', values='score')
            
            accident_day_score_diff = pd.DataFrame(
                data = accident_day_score_pivot.values[:, :-1] - accident_day_score_pivot.values[:, 1:],
                index = accident_day_score_pivot.index,
                columns = accident_day_score_pivot.columns[:-1]
            )
            accident_day_score_diff.columns = accident_day_score_diff.columns.map(lambda x: f"diff_{x}to{x+1}")

            day_count = 0
            is_first = True
            for i in range(1, 5):
                other_day_score = pd.read_csv(f"score/other_day_score{i}.csv")
                day_count += 1
                other_day_score = other_day_score[['link_id', 'time_bin_index', 'score']]
                other_day_score_pivot = other_day_score.pivot(index='link_id', columns='time_bin_index', values='score')

                # 10분 단위 점수 차이 계산
                other_day_score_diff = pd.DataFrame(
                    data = other_day_score_pivot.values[:, :-1] - other_day_score_pivot.values[:, 1:],
                    index = other_day_score_pivot.index,
                    columns = other_day_score_pivot.columns[:-1]
                ).fillna(0)
                
                other_day_score_diff.columns = other_day_score_diff.columns.map(lambda x: f"diff_{x}to{x+1}")
                if is_first:
                    other_day_score_mean = other_day_score_diff
                    is_first = False
                else:
                    other_day_score_mean = other_day_score_mean + other_day_score_diff # 값 누적 합
            # 평균 계산
            other_day_score_mean /= day_count
            other_day_score_mean = other_day_score_mean.replace(0, np.nan)

            result = get_results_score(accident_day_score_diff, other_day_score_mean)

        else:
            # 둘 다 pivot 테이블
            accident_day_score_diff = self.get_accident_day_score_diff(save_score=True) 
            other_day_score_mean = self.get_other_day_score_diff_mean(save_score=True)
            result = get_results_score(accident_day_score_diff, other_day_score_mean)

        return result

    def run(self):
        result = self.mapmatching_result(from_saved=False)
        result.to_csv('result/result_link_score.csv', encoding='cp949')
        result = result.sort_values(by="result_score", ascending=False)

        print('<매칭 결과>')
        for i in range(len(result)):
            if i > 4:
                break
            result_link = result.iloc[i]
            print(f'{i+1}순위 후보 링크 id: {result_link.name}')
            
        
def main():
    tass_data_path = 'tass_dataset/taas_new.csv'
    ps_data_path = 'traj_sample/alltraj_20231211.txt'
    moct_network_path = 'moct_link/link'

    accidentlinkmatching = AccidentMapMatchingProcessor(tass_data_path, ps_data_path, moct_network_path)
    accidentlinkmatching.run()

if __name__ == '__main__':
    main()
