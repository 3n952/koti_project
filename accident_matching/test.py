import geopandas as gpd
import pandas as pd

from accident_matching import AccidentMatching, AccidentDataPreprocessing


class AccidentMapMatchingProcess(AccidentDataPreprocessing, AccidentMatching):
    def __init__(self, tass_data_path, ps_data_path, moct_network_path):
        AccidentDataPreprocessing.__init__(self, tass_data_path, ps_data_path,moct_network_path)
        AccidentMatching.__init__(self, radius=300)
        self.preparing_dataset()

    def preparing_dataset(self):
        '''
        3개의 data를 accidentdatapreprocessing 클래스로 전처리하여 지역, 시간 필터링 df를 산출하는 함수
        
        returns: 
            self.accident_day_df: 사고 지점 근처의 후보 링크에 대한 궤적 데이터
        '''

        self.accident_day_df = gpd.GeoDataFrame(columns=['link_id', 'trip_id', 'pointTime', 'speed', 'geometry', 'time_bin_index'], 
                             geometry='geometry', crs="EPSG:5179")
        
        for ps_near_chunk in self.link_ps_merge_sampling():
            self.candidate_links_with_timebin(
                tass_sample_gdf=self.tass_sample_gdf, 
                moct_link_gdf=self.moct_link_gdf, 
                link_ps_merge_near_acctime_gdf=ps_near_chunk
            )
            self.accident_day_df = pd.concat([self.accident_day_df, self.ps_on_candidate_links_with_timebin_gdf], ignore_index=True)

    def compare_score_other_days(self):

        self.remain_days_traj(preprocessing_on_local=True)

    def run(self):
        sample = next(self.remain_days_traj(preprocessing_on_local=True))
        sample.head()

        # # 사고 시간이 해당되는 날에 대한 link score
        # self.candidate_link_score()

def main():
    tass_data_path = 'tass_dataset/tass_2019_to_2023_241212.csv'
    ps_data_path = 'traj_sample/alltraj_20231211.txt'
    moct_network_path = 'moct_link/link'

    accidentlinkmatching = AccidentMapMatchingProcess(tass_data_path, ps_data_path, moct_network_path)
    accidentlinkmatching.run()

if __name__ == '__main__':
    main()
