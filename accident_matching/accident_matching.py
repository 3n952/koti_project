import pandas as pd
import numpy as np
from datetime import datetime
import geopandas as gpd
from shapely.geometry import Point, LineString
from collections import defaultdict
from utils import dt2unix, ymd_spliter, get_weekdays_in_same_week, corresponding_sample_timecode_unixtime


class AccidentDataPreprocessing:
    '''
    사고 - 맵 매칭을 위한 사고 데이터, PS 포인트 데이터, MOCT link data를 pandas 혹은 geopandas의 dataframe으로 변환 클래스

    attributes:
        tass_data_path: 사고 데이터 로컬 파일 경로
        ps_data_path: ps 포인트 데이터 로컬 파일 경로
        moct_network_path: moct link 데이터 로컬 파일 경로
    '''

    def __init__(self, tass_data_path, ps_data_path, moct_network_path, preprocessing_on_local = True):
        self.tass_data_path = tass_data_path
        self.ps_data_path = ps_data_path
        self.moct_network_path = moct_network_path
        self.init_moct_network2gdf()
        self.init_tass_data2gdf()

    def init_moct_network2gdf(self):
        moct_network_link_gdf = gpd.read_file(self.moct_network_path, encoding='euc-kr')
        moct_network_link_gdf = moct_network_link_gdf[['link_id', 'road_rank', 'sido_id', 'sgg_id','geometry']]
        moct_network_link_gdf.set_crs(epsg=5179, inplace=True)

        # 지역 필터링 (서울시)
        self.moct_link_gdf = moct_network_link_gdf[moct_network_link_gdf['sido_id'] == 11000]
    
    def ps_data2gdf(self, chunksize=10000000):
        '''
        ps 데이터를 CSV로 청크단위로 기본 전처리하여 gdf를 제네레이터로 반환

            :param chunksize: 청크 단위 사이즈
            :yield: 기본 전처리된 ps의 gdf타입의 데이터
        '''
        for ps_dataset in pd.read_csv(self.ps_data_path, 
                                  names = ['trip_id', 'reliability2', 'mathcingRate', 'pointTime', 
                                           'pointX', 'pointY', 'link_id', 'speed'], chunksize=chunksize):
    
            # 좌표를 geometry로 변환
            ps_dataset['geometry'] = gpd.points_from_xy(ps_dataset['pointX'], ps_dataset['pointY'])
            # GeoDataFrame 변환
            ps_gdf = gpd.GeoDataFrame(ps_dataset, geometry='geometry', crs="EPSG:5179")
            # 필터링: link_id가 -1이 아닌 경우만 사용
            ps_gdf.loc[:] = ps_gdf[ps_gdf['link_id'] != -1]
            # link_id를 문자열로 변환
            ps_gdf['link_id'] = ps_gdf['link_id'].astype(str)

            yield ps_gdf  # 제네레이터로 반환

            # self.ps_gdf = gpd.GeoDataFrame(ps_dataset, geometry=gpd.points_from_xy(ps_dataset['pointX'],
            #                                                                         ps_dataset['pointY']), 
            #                                                                         crs="EPSG:5179")
            # self.ps_gdf = self.ps_gdf[self.ps_gdf['link_id'] != -1]
            # self.ps_gdf['link_id'] = self.ps_gdf['link_id'].astype(str)

            # yield self.ps_gdf
    
    def init_tass_data2gdf(self):
        tass_dataset = pd.read_csv(self.tass_data_path)
        tass_df = tass_dataset[['acdnt_no', 'acdnt_dd_dc', 'occrrnc_time_dc', 'legaldong_name', 'x_crdnt_crdnt',
                                        'y_crdnt_crdnt', 'acdnt_gae_dc', 'wrngdo_vhcle_asort_dc', 
                                        'dmge_vhcle_asort_dc', 'road_div']]
        # 원본 수정을 위한 copy 
        tass_df = tass_df.copy()
        tass_df.loc[:, 'occrrnc_time_dc'] = tass_df['occrrnc_time_dc'].str.replace('시', '', regex=True)
        tass_df.loc[:, 'legaldong_name'] = tass_df['legaldong_name'].apply(
            lambda x: next((district for district in x.split() if '시' in district), '')
            )
        
        # 날짜와 시간을 결합하여 datetime 형식 변환
        tass_df.loc[:, 'datetime'] = pd.to_datetime(tass_df['acdnt_dd_dc'] + ' ' + 
                                                            tass_df['occrrnc_time_dc'] + ':00:00') # unixtime 변환 (초 단위)
        tass_df.loc[:, 'unixtime']  = tass_df['datetime'].astype('int64') // 10**9  # 초 단위로 변환
        tass_df.loc[:, 'unixtime']  = tass_df['unixtime'] - 9 * 3600 # KST 기준 맞추기

        # 조건 필터링 - 서울 / 중상,사망 사고 / 승용, 승합, 화물
        tass_df = tass_df[tass_df['legaldong_name'] == '서울특별시']
        tass_df = tass_df[(tass_df['acdnt_gae_dc'].isin(['중상사고', '사망사고']))]
        self.tass_df = tass_df[(tass_df['wrngdo_vhcle_asort_dc'].isin(['승용', '승합', '화물'])) | 
                                          (tass_df['dmge_vhcle_asort_dc'].isin(['승용', '승합', '화물']))]

        self.tass_sampling()

    def tass_sampling(self):
        ymd = self.ps_data_path.split('/')[1].split('.')[0][-8:]
        y,m,d = ymd_spliter(ymd)
        # 시간 필터링 -> 비교 moct dtg가 
        # 하루에 대한 값
        self.tass_df = self.tass_df[self.tass_df['unixtime'].between(dt2unix(f'{y}-{m}-{d} 00:00:00'), 
                                                                              dt2unix(f'{y}-{m}-{d} 23:59:59'))]
        # 특정 사고 랜덤 추출
        tass_sample_df = self.tass_df.sample(n=1)
        self.tass_sample_gdf = gpd.GeoDataFrame(tass_sample_df, 
                                          geometry=gpd.points_from_xy(tass_sample_df['x_crdnt_crdnt'], 
                                                                      tass_sample_df['y_crdnt_crdnt']), 
                                                                      crs="EPSG:5179")
        
        tass_sample_timestamp = self.tass_sample_gdf['unixtime']
        tass_sample_timecode = self.tass_sample_gdf['occrrnc_time_dc'].astype(int)

        uptime = tass_sample_timestamp + 5400
        downtime = tass_sample_timestamp - 1800

        self.uptime = uptime.iloc[0]  
        self.downtime = downtime.iloc[0]
        self.timecode = tass_sample_timecode
    
    def redefine_timerange(self, unixtime):
        uptime = unixtime + 5400
        downtime = unixtime - 1800

        self.uptime = uptime.iloc[0]  
        self.downtime = downtime.iloc[0]
        self.timecode = unixtime

    def merge_link_ps(self):
        '''
        국내 전체에 분포된 ps 포인트를 관심영역을 지정한 link에 매칭
        '''
        # common_link_ids = set(self.ps_gdf['link_id']) & set(self.moct_link_gdf['link_id'])

        # if len(common_link_ids) > 0:
        #     self.link_ps_merge_gdf = pd.merge(self.ps_gdf, self.moct_link_gdf, on='link_id', how='inner')
        
        # # 결합 후 필요 컬럼 남기기 + 이름 바꾸기
        # self.link_ps_merge_gdf = self.link_ps_merge_gdf[['link_id', 'trip_id', 'pointTime','speed', 'geometry_x']]
        # self.link_ps_merge_gdf = self.link_ps_merge_gdf.rename(columns={'geometry_x': 'geometry'})

        for ps_chunk_df in self.ps_data2gdf():
            link_ps_merge_gdf = pd.merge(ps_chunk_df, self.moct_link_gdf, on='link_id', how='inner')
            link_ps_merge_gdf = link_ps_merge_gdf[['link_id', 'trip_id', 'pointTime','speed', 'geometry_x']]
            link_ps_merge_gdf = link_ps_merge_gdf.rename(columns={'geometry_x': 'geometry'})
            
            yield link_ps_merge_gdf

    def link_ps_merge_sampling(self):
        '''
        merge_link_ps의 df에서 사고 시간단위의 앞뒤 30분을 필터링
        '''
        for link_ps_merge_chunk_gdf in self.merge_link_ps():
            link_ps_merge_near_acctime_gdf = link_ps_merge_chunk_gdf[link_ps_merge_chunk_gdf['pointTime'].between(self.downtime, self.uptime)]

            yield link_ps_merge_near_acctime_gdf
    
    def total_link_ps_df(self):
        self.total_link_ps_df = None
        for link_ps_chunk in self.link_ps_merge_sampling():
            pass

    
    def remain_days_traj(self, preprocessing_on_local, remain_ps_data_path=None):
        '''
        평일 or 주말에 대한 trajectory를 반환하는 제너레이터 함수

            :param remain_ps_data_path: 병합할 추가 데이터 경로
            :return: 병합된 GeoDataFrame
        '''
        # 사고가 난 날 이외의 날에 대한 trajectory를 로컬에서 처리하는 경우
        if preprocessing_on_local:
            input_date_str = self.ps_data_path.split('/')[1].split('_')[:-5]
            daystr_ofweeks = get_weekdays_in_same_week(input_date_str) # datetime 반환

            for date in daystr_ofweeks:
                new_date_str = date.strftime("%Y%m%d")
                self.ps_data_path = f'traj_sample/alltraj_{new_date_str}.txt'
                new_unixtime = corresponding_sample_timecode_unixtime(new_date_str, self.timecode)
                self.redefine_timerange(new_unixtime)

                for link_ps_merge_near_acctime_gdf_chunk in self.link_ps_merge_sampling():
                    yield link_ps_merge_near_acctime_gdf_chunk
        
        else:
            remain_ps_data= pd.read_csv(remain_ps_data_path, 
                                  names = ['trip_id', 'reliability2', 'mathcingRate', 'pointTime', 
                                           'pointX', 'pointY', 'link_id', 'speed'])
            remain_ps_df = gpd.GeoDataFrame(remain_ps_data, geometry=gpd.points_from_xy(remain_ps_data['pointX'],
                                                                                remain_ps_data['pointY']), 
                                                                                crs="EPSG:5179")
            remain_ps_df = remain_ps_df[remain_ps_df['link_id'] != -1]
            remain_ps_df['link_id'] = remain_ps_df['link_id'].astype(str)

            daystr_ofweeks = get_weekdays_in_same_week(input_date_str)
            for date in daystr_ofweeks:
                new_date_str = date.strftime("%Y%m%d")
                self.ps_data_path = f'traj_sample/alltraj_{new_date_str}.txt'
                new_ps_gdf = self.ps_data2gdf()
                self.merge_remain_days_df = pd.concat([self.merge_remain_days_df, new_ps_gdf], ignore_index=True)
            
            return self.merge_remain_days_df

class AccidentMatching():

    def __init__(self, radius):
        self.radius = radius
    
    def extract_link_candidate(self, tass_sample_gdf, moct_link_gdf):
        final_candidate_links = defaultdict(list)

        def find_link_candidate(accident_point: Point, road_gdf: gpd.GeoDataFrame):
            """
            사고 위치와 가까운 도로 링크 후보 찾는 함수 (도로 전체가 반경 내에 존재하는지 확인)

                :param accident_point: 사고 위치 (shapely.geometry.Point)
                :param road_gdf: 도로 네트워크 GeoDataFrame (geometry 열 포함)
                :param radius: 후보 추출 반경 (미터 단위)

                :return: {link_id: LineString} 구조의 딕셔너리 반환
            """
            candidate_links = {}

            for _, row in road_gdf.iterrows():
                segment = row['geometry']
                link_id = row['link_id']  # link_id 추출

                if segment.is_empty:
                    continue

                # 도로(링크) 전체가 반경 내에 포함되는지 검사
                # accident point는 하나의 사고 포인트
                if segment.distance(accident_point) <= self.radius:  
                    candidate_links[link_id] = segment  # {link_id: LineString} 형태로 저장

            return candidate_links

        # apply() 결과를 defaultdict로 병합
        tass_sample_gdf['geometry'].apply(
            lambda point: final_candidate_links.update(find_link_candidate(point, moct_link_gdf)))       

        return final_candidate_links

    def candidate_links_with_timebin(self, tass_sample_gdf, moct_link_gdf, link_ps_merge_near_acctime_gdf):
        final_candidate_links = self.extract_link_candidate(tass_sample_gdf, moct_link_gdf)

        candidate_links = set(final_candidate_links.keys())
        self.candidate_links_gdf = moct_link_gdf[moct_link_gdf['link_id'].isin(candidate_links)]
        self.ps_on_candidate_links_gdf = link_ps_merge_near_acctime_gdf[link_ps_merge_near_acctime_gdf['link_id'].isin(candidate_links)]
        
        sample_time = tass_sample_gdf['unixtime'].values
        ps_on_candidate_links_gdf = self.ps_on_candidate_links_gdf.copy() 
        # 10분 간격으로 몇 번째 구간(인덱스)에 속하는지 계산
        ps_on_candidate_links_gdf['time_bin_index'] = ((ps_on_candidate_links_gdf['pointTime'] - sample_time) // 600).astype(int)  # 10분(600초) 단위
        self.ps_on_candidate_links_with_timebin_gdf = ps_on_candidate_links_gdf[ps_on_candidate_links_gdf['time_bin_index'].between(-3,9)]

    def candidate_link_score(self):

        def nearest_quantile_95(series):
            sorted_values = series.sort_values().values  # 정렬
            index = int(len(sorted_values) * 0.95)  # 95% 위치 찾기
            return sorted_values[min(index, len(sorted_values) - 1)]  # 가장 가까운 인덱스 값 반환
        
        unique_ps_on_candidate_links_with_timebin_gdf = self.ps_on_candidate_links_with_timebin_gdf\
                                                        .sort_values(by=['link_id', 'trip_id', 'speed'])\
                                                        .groupby(['link_id', 'trip_id', 'time_bin_index']).first().reset_index()
        
        # 2시간 단위 기준 속도 95% 백분위수 계산
        link_percentile95_df = unique_ps_on_candidate_links_with_timebin_gdf.groupby(['link_id']).agg(
                                                                    percentile95=('speed', nearest_quantile_95)  # 95th Percentile 계산
                                                                    ).reset_index()

        # 10분 단위 기준 속도 중위값 계산
        link_median_df = unique_ps_on_candidate_links_with_timebin_gdf.groupby(['link_id', 'time_bin_index']).agg(median=('speed', 'median')).reset_index()
        
        self.link_score_df = link_median_df.merge(link_percentile95_df, on='link_id', how='left')
        self.link_score_df['score'] = self.link_score_df['median'] / self.link_score_df['percentile95']
   

