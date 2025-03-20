import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import logging
import math 
import numpy as np
from utils import get_weekdays_in_same_week

from accident_matching import AccidentMatching, AccidentDataPreprocessing

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class AccidentMapMatchingProcess(AccidentDataPreprocessing, AccidentMatching):
    def __init__(self, tass_data_path, ps_data_path, moct_network_path):
        AccidentDataPreprocessing.__init__(self, tass_data_path, ps_data_path, moct_network_path)
        AccidentMatching.__init__(self, radius=300)

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

    def get_other_days_score(self):
        '''
        3개의 data(tass, ps, moct)를 accidentdatapreprocessing 클래스로 전처리하여 지역, 시간 필터링 df를 산출하는 함수

        returns:
            self.other_day_df: 사고 당일을 제외한 평일 or 주말의 후보 링크 + 시간대에 대한 궤적 데이터
        '''
        input_date_str = self.ps_data_path.split('/')[1].split('_')[1][:-4] # 사고 날짜 ps_data_path
        daystr_ofweeks = get_weekdays_in_same_week(input_date_str) # datetime 반환
        for idx_day, date in enumerate(daystr_ofweeks):
            # 하루치 df 초기화
            self.other_day_df = gpd.GeoDataFrame(columns=['link_id', 'trip_id', 'pointTime', 'speed', 'geometry', 'time_bin_index'], 
                             geometry='geometry', crs="EPSG:5179")
            new_date_str = date.strftime("%Y%m%d")
            self.ps_data_path = rf'D:\traj_samples\alltraj_{new_date_str}.txt'
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
            other_day_score.to_csv(f'scores/other_day_score{idx_day+1}.csv', index=False)
            logging.info(f"사고 이외 날짜 기반 {idx_day+1}번째 날 소통 점수 산출 완료")
            yield other_day_score

    def get_accident_day_score_diff(self):
        self.accident_day_dataloader()
        #accident_day_df = self.accident_day_df # 'link_id', 'trip_id', 'pointTime', 'speed', 'geometry', 'time_bin_index'
        logging.info(f"사고 당일 데이터 기반 소통 점수 산출 중... ")
        accident_day_score = self.candidate_link_score(ps_on_candidate_links_gdf=self.accident_day_df)
        accident_day_score = accident_day_score[['link_id', 'time_bin_index', 'score']]
        accident_day_score_pivot = accident_day_score.pivot(index='link_id', columns='time_bin_index', values='score')
        # 다음 10분 점수와의 차이 계산 
        accident_day_score_pivot_diff = (accident_day_score_pivot.iloc[:, :-1] - accident_day_score_pivot.iloc[:, 1:]).where(accident_day_score_pivot.iloc[:, :-1].notna() & accident_day_score_pivot.iloc[:, 1:].notna())
        print(accident_day_score_pivot_diff.head())
        print(accident_day_score_pivot_diff.columns)
        accident_day_score_pivot_diff = accident_day_score_pivot_diff.reset_index() # index = [link_id', -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        print(accident_day_score_pivot_diff.head())
        print(accident_day_score_pivot_diff.columns)

        return accident_day_score_pivot_diff
    
    def get_other_day_score_diff_mean(self):
        other_day_mean_df = pd.DataFrame(columns=['link_id', -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        
        for other_day_score_per_oneday in self.get_other_days_score():
            other_day_score_per_oneday = other_day_score_per_oneday[['link_id', 'time_bin_index', 'score']]
            other_day_score_per_oneday_pivot = other_day_score_per_oneday.pivot(index='link_id', columns='time_bin_index', values='score')
            other_day_score_per_oneday_pivot_diff = (other_day_score_per_oneday_pivot.iloc[:, :-1] - other_day_score_per_oneday_pivot.iloc[:, 1:]).where(other_day_score_per_oneday_pivot.iloc[:, :-1].notna() & other_day_score_per_oneday_pivot.iloc[:, 1:].notna())
            other_day_score_per_oneday_pivot_diff = other_day_score_per_oneday_pivot_diff.reset_index()

            other_day_mean_df = pd.concat([other_day_mean_df, other_day_score_per_oneday_pivot_diff])

        other_day_mean_df = other_day_mean_df.fillna(0).groupby(level=0).mean()
        print(other_day_mean_df.head())
        logging.info(f"사고 이외 날짜 소통 점수 평균 산출 완료")

        return other_day_mean_df

    def get_candidate_link_scores(self, is_save=False):
        self.accident_day_dataloader()
        logging.info(f"사고 당일 데이터 기반 소통 점수 산출 중... ")
        accident_day_score = self.candidate_link_score(ps_on_candidate_links_gdf=self.accident_day_df)
        accident_day_score = accident_day_score[['link_id', 'time_bin_index', 'score']]
        logging.info(f"사고 당일 데이터 기반 소통 점수 산출 완료 ")

        logging.info(f"사고 이외 날짜 전체 소통 점수 산출 시작")
        other_day_score = self.other_days_score()
        other_day_score = other_day_score[['link_id', 'time_bin_index', 'score']]
        logging.info(f"사고 이외 날짜 전체 소통 점수 산출 완료 ")

        # if is_save:
        #     accident_day_score.to_csv('scores/accident_day_score.csv', index=False)
        #     other_day_score.to_csv('scores/other_day_score.csv', index=False)

        return accident_day_score, other_day_score

    def run(self, is_save=False):
        accident_day_score_diff = self.get_accident_day_score_diff()
        other_day_mean_df = self.get_other_day_score_diff_mean()
        if is_save:
            accident_day_score_diff.to_csv('scores/accident_day_score_diff.csv', index=False)
            other_day_mean_df.to_csv('scores/other_day_mean.csv', index=False)

        # result = ((df_after - df_before) / df_before) * 100  # 백분율 변화량
        # print("\n=== 변화율 (%) ===")
        # print(df_ratio)

def main():
    tass_data_path = 'tass_dataset/tass_2019_to_2023_241212.csv'
    ps_data_path = 'traj_sample/alltraj_20231211.txt'
    moct_network_path = 'moct_link/link'

    accidentlinkmatching = AccidentMapMatchingProcess(tass_data_path, ps_data_path, moct_network_path)
    accidentlinkmatching.run()
    print('over')
if __name__ == '__main__':
    main()
