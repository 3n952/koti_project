import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import logging
import math 

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
                tass_sample_gdf=self.tass_sample_gdf, 
                link_ps_merge_near_acctime_gdf=ps_near_chunk
            )
            self.accident_day_df = pd.concat([self.accident_day_df, ps_on_candidate_links_with_timebin_gdf], ignore_index=True)

        logging.info(f"사고 당일 데이터 로드 완료. 결과: 총 {len(self.accident_day_df)}개 데이터 포인트 추출")

    def other_days_score(self):
        '''
        3개의 data(tass, ps, moct)를 accidentdatapreprocessing 클래스로 전처리하여 지역, 시간 필터링 df를 산출하는 함수

        returns:
            self.other_day_df: 사고 당일을 제외한 평일 or 주말의 후보 링크 + 시간대에 대한 궤적 데이터
        '''
        self.other_day_df = gpd.GeoDataFrame(columns=['link_id', 'trip_id', 'pointTime', 'speed', 'geometry', 'time_bin_index'], 
                             geometry='geometry', crs="EPSG:5179")
        total_other_day_score = pd.DataFrame(columns=['link_id', 'time_bin_index', 'median', 'percentile95', 'score'])

        input_date_str = self.ps_data_path.split('/')[1].split('_')[1][:-4] # 사고 날짜 ps_data_path
        daystr_ofweeks = get_weekdays_in_same_week(input_date_str) # datetime 반환
        for date in daystr_ofweeks:
            new_date_str = date.strftime("%Y%m%d")
            self.ps_data_path = rf'D:\traj_samples\alltraj_{new_date_str}.txt'
            total_chunk_size = self.get_data_size()
        
            for idx_day, other_ps_chunk in tqdm(enumerate(self.remain_days_traj(new_date_str, preprocessing_on_local=True),
                                   total=total_chunk_size, 
                                   desc="사고 이외 날짜 데이터(gps point) 로드 중..", 
                                   unit="per chunk size 10000000")):
                ps_on_candidate_links_with_timebin_gdf = self.candidate_links_with_timebin(
                    tass_sample_gdf=self.tass_sample_gdf,  
                    link_ps_merge_near_acctime_gdf=other_ps_chunk
                )
                self.other_day_df = pd.concat([self.other_day_df, ps_on_candidate_links_with_timebin_gdf], ignore_index=True)
                
                logging.info(f"사고 이외 날짜 데이터 기반 {idx_day+1}번째 날 소통 점수 산출 중...")
                other_day_score = self.candidate_link_score(ps_on_candidate_links_gdf=self.other_day_df)
                total_other_day_score = pd.concat([total_other_day_score, other_day_score], ignore_index=True)


        logging.info(f"사고 이외 날짜 데이터 로드 완료. 결과: 총 {len(self.other_day_df)}개 데이터 포인트 추출")
        return total_other_day_score

    def compare_score_with_other_days(self):
        self.accident_day_dataloader()
        logging.info(f"사고 당일 데이터 기반 소통 점수 산출 중... ")
        accident_day_score = self.candidate_link_score(ps_on_candidate_links_gdf=self.accident_day_df)
        accident_day_score = accident_day_score[['link_id', 'time_bin_index', 'score']]
        logging.info(f"사고 당일 데이터 기반 소통 점수 산출 완료 ")

        logging.info(f"사고 이외 날짜 데이터 기반 소통 점수 산출 중... ")
        other_day_score = self.other_days_score()
        other_day_score = other_day_score[['link_id', 'time_bin_index', 'score']]
        logging.info(f"사고 이외 날짜 데이터 기반 소통 점수 산출 완료 ")

        accident_day_score.to_csv('accident_day_score.csv', index=False)
        other_day_score.to_csv('other_day_score.csv', index=False)

    def run(self):
        # dataload
        self.compare_score_with_other_days()


def main():
    tass_data_path = 'tass_dataset/tass_2019_to_2023_241212.csv'
    ps_data_path = 'traj_sample/alltraj_20231211.txt'
    moct_network_path = 'moct_link/link'

    accidentlinkmatching = AccidentMapMatchingProcess(tass_data_path, ps_data_path, moct_network_path)
    accidentlinkmatching.run()
    print('test')
if __name__ == '__main__':
    main()
