import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from collections import defaultdict
import logging
from typing import Generator

from utils import dt2unix, ymd_spliter, corresponding_sample_timecode_unixtime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class AccidentDataPreprocessing:
    '''
    사고 - 맵 매칭을 위한 사고 데이터, PS 포인트 데이터, MOCT link data를 pandas 혹은 geopandas의 dataframe으로 변환 클래스
    attributes:
        taas_data_path: 사고 데이터 로컬 파일 경로
        ps_data_path: ps 포인트 데이터 로컬 파일 경로
        moct_network_path: moct link 데이터 로컬 파일 경로
    '''

    def __init__(self, taas_data_path, ps_data_path, moct_network_path):
        self.taas_data_path = taas_data_path
        self.ps_data_path = ps_data_path
        self.moct_network_path = moct_network_path
        # TODO: 네트워크 반복 로드 문제 해결하기, 초기화 과정 바꾸기 ex. self.moct_network_gdf = self.load_network(moct_network_path)
        self.init_moct_network2gdf()
        self.init_taas_data2gdf()

    def init_moct_network2gdf(self):
        logging.info("MOCT 네트워크 데이터 로드 중..")
        moct_network_link_gdf = gpd.read_file(self.moct_network_path, encoding='euc-kr')
        moct_network_link_gdf = moct_network_link_gdf[['link_id', 'road_rank', 'sido_id', 'sgg_id', 'geometry']]
        moct_network_link_gdf.set_crs(epsg=5179, inplace=True)
        self.moct_link_gdf = moct_network_link_gdf
        logging.info("MOCT 네트워크 데이터 로드 완료")

    # TODO: taas 경우 / 돌발 상황 정보의 경우 구분하기 -> 주석처리된 부분 처리
    def init_taas_data2gdf(self):
        logging.info("사고 데이터(taas) 로드 중..")
        try:
            taas_dataset = pd.read_csv(self.taas_data_path, encoding='cp949')
        except UnicodeDecodeError:
            taas_dataset = pd.read_csv(self.taas_data_path, encoding='UTF-8')

        taas_df = taas_dataset[['acdnt_dd_dc', 'occrrnc_time_dc', 'x_crdnt_crdnt', 'y_crdnt_crdnt']]
        taas_df.loc[:, 'occrrnc_time_dc'] = taas_df['occrrnc_time_dc'].str.replace('시', '', regex=True)  # timecode

        # 날짜와 시간을 결합하여 datetime 형식 변환
        # unixtime 변환
        taas_df.loc[:, 'datetime'] = pd.to_datetime(taas_df['acdnt_dd_dc'] + ' ' + taas_df['occrrnc_time_dc'] + ':00:00')
        # KST 기준 맞추기
        taas_df.loc[:, 'unixtime'] = taas_df['datetime'].astype('int64') // 10**9 - (9 * 3600)

        self.taas_df = taas_df
        logging.info("사고 데이터(taas) 로드 완료")
        self.taas_sampling()
        logging.info(f"사고 데이터(taas) 샘플 추출 완료\n<====================사고 샘플 데이터 info====================>\n{self.taas_sample_gdf.head()}")

    def ps_data2gdf(self, chunksize=1000000):
        '''
        ps 데이터를 CSV로 청크단위로 기본 전처리하여 gdf를 제네레이터로 반환
            :param chunksize: 청크 단위 사이즈
            :yield: 기본 전처리된 ps의 gdf타입의 데이터
        '''
        columns = ['trip_id', 'pointTime', 'pointX', 'pointY', 'link_id', 'speed']
        for ps_dataset in pd.read_csv(self.ps_data_path, names=columns, chunksize=chunksize):
            # 좌표를 geometry로 변환
            ps_dataset['geometry'] = gpd.points_from_xy(ps_dataset['pointX'], ps_dataset['pointY'])
            # GeoDataFrame 변환
            ps_gdf = gpd.GeoDataFrame(ps_dataset, geometry='geometry', crs="EPSG:5179")
            # 필터링: link_id가 -1이 아닌 경우만 사용
            ps_gdf.loc[:] = ps_gdf[ps_gdf['link_id'] != -1]
            # link_id를 문자열로 변환
            ps_gdf['link_id'] = ps_gdf['link_id'].astype(str)

            yield ps_gdf  # 제네레이터로 반환

    # TODO: 랜덤 샘플링 추출 / 특정 인덱스 샘플링 추출 기능 복합적으로 구현하기
    def taas_sampling(self, seed=33):
        ymd = self.ps_data_path.split('/')[1].split('.')[0][-8:]
        y, m, d = ymd_spliter(ymd)
        # 시간 필터링
        # 하루에 대한 값
        self.taas_df = self.taas_df[self.taas_df['unixtime'].between(dt2unix(f'{y}-{m}-{d} 00:00:00'),
                                                                     dt2unix(f'{y}-{m}-{d} 23:59:59'))]
        # 특정 사고 랜덤 샘플링
        taas_sample_df = self.taas_df.sample(n=1, random_state=seed)

        # 특정 사고 인덱스 기반 샘플링
        # taas_sample_df = self.taas_df.iloc[[858]]

        self.taas_sample_gdf = gpd.GeoDataFrame(taas_sample_df,
                                                geometry=gpd.points_from_xy(taas_sample_df['x_crdnt_crdnt'],
                                                                            taas_sample_df['y_crdnt_crdnt']),
                                                crs="EPSG:5179")
        self.taas_sample_gdf.to_csv('result/taas_sample.csv')
        taas_sample_timestamp = self.taas_sample_gdf['unixtime'].iloc[0]
        taas_sample_timecode = self.taas_sample_gdf['occrrnc_time_dc'].astype(int).iloc[0]
        self.uptime = taas_sample_timestamp + 5400
        self.downtime = taas_sample_timestamp - 1800
        self.timecode = taas_sample_timecode

    def redefine_timerange(self, unixtime):
        self.uptime = unixtime + 5400
        self.downtime = unixtime - 1800
        self.new_timestamp = unixtime

    def merge_link_ps(self):
        '''
        국내 전체에 분포된 ps 포인트를 network link에 매칭
        '''
        for ps_chunk_df in self.ps_data2gdf():
            link_ps_merge_gdf = pd.merge(ps_chunk_df, self.moct_link_gdf, on='link_id', how='inner')
            link_ps_merge_gdf = link_ps_merge_gdf[['link_id', 'trip_id', 'pointTime', 'speed', 'geometry_x']]
            link_ps_merge_gdf = link_ps_merge_gdf.rename(columns={'geometry_x': 'geometry'})
            yield link_ps_merge_gdf

    def link_ps_merge_sampling(self):
        '''
        merge_link_ps의 df에서 사고 시간단위의 앞뒤 30분을 필터링
        '''
        for link_ps_merge_chunk_gdf in self.merge_link_ps():
            link_ps_merge_near_acctime_gdf = link_ps_merge_chunk_gdf[link_ps_merge_chunk_gdf['pointTime'].between(self.downtime, self.uptime)]
            yield link_ps_merge_near_acctime_gdf

    def remain_days_traj(self, new_date_str, preprocessing_on_local) -> Generator[gpd.GeoDataFrame, None, None]:
        '''
        평일 or 주말에 대한 trajectory를 반환하는 제너레이터 함수
        Args:
            preprocessing_on_local: 로컬에서 trajectory를 반환 여부 결정
            remain_ps_data_path: koti에서 trajectory를 반환한 데이터 경로
        returns
            trajectory generator
        '''
        # 사고가 난 날 이외의 날에 대한 trajectory를 로컬에서 처리하는 경우
        if preprocessing_on_local:
            new_unixtime = corresponding_sample_timecode_unixtime(new_date_str, self.timecode)
            self.redefine_timerange(new_unixtime)
            for link_ps_merge_near_acctime_gdf_chunk in self.link_ps_merge_sampling():
                yield link_ps_merge_near_acctime_gdf_chunk
        else:
            print('add code')


class AccidentMatching:
    def __init__(self, radius):
        self.radius = radius

    def extract_link_candidate(self, taas_sample_gdf, moct_link_gdf):
        final_candidate_links = defaultdict(list)

        def find_link_candidate(accident_point: Point, road_gdf: gpd.GeoDataFrame):
            """
            사고 위치와 가까운 도로 링크 후보 찾는 함수 (도로 전체가 반경 내에 존재하는지 확인)
            Args:
                accident_point: 사고 위치 (shapely.geometry.Point)
                road_gdf: 도로 네트워크 GeoDataFrame (geometry 열 포함)
            return:
                {link_id: LineString} 구조의 딕셔너리 반환
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
        taas_sample_gdf['geometry'].apply(
            lambda point: final_candidate_links.update(find_link_candidate(point, moct_link_gdf)))
        return final_candidate_links

    def get_candidate_links(self, taas_sample_gdf, moct_link_gdf):
        final_candidate_links = self.extract_link_candidate(taas_sample_gdf, moct_link_gdf)
        self.candidate_links = set(final_candidate_links.keys())

    def candidate_links_with_timebin(self, new_timestamp, link_ps_merge_near_acctime_gdf):
        self.ps_on_candidate_links_gdf = link_ps_merge_near_acctime_gdf[link_ps_merge_near_acctime_gdf['link_id'].isin(self.candidate_links)]
        # TODO: copy()필요성 검증하기: 메모리 누수 문제
        ps_on_candidate_links_gdf = self.ps_on_candidate_links_gdf.copy()

        # 10분 간격으로 몇 번째 구간(인덱스)에 속하는지 계산
        ps_on_candidate_links_gdf['time_bin_index'] = ((ps_on_candidate_links_gdf['pointTime'] - new_timestamp) // 600).astype(int)  # 10분(600초) 단위
        ps_on_candidate_links_gdf[ps_on_candidate_links_gdf['time_bin_index'].between(-3, 9)]

        return ps_on_candidate_links_gdf

    def candidate_link_score(self, ps_on_candidate_links_gdf):
        '''
        후보링크 기준 시간 단위별 소통 score 산출
        '''
        def nearest_quantile_95(series):
            sorted_values = series.sort_values().values  # 정렬
            index = int(len(sorted_values) * 0.95)  # 95% 위치 찾기
            return sorted_values[min(index, len(sorted_values) - 1)]  # 가장 가까운 인덱스 값 반환

        unique_ps_on_candidate_links_with_timebin_gdf = ps_on_candidate_links_gdf.sort_values(by=['link_id', 'trip_id', 'speed']).\
            groupby(['link_id', 'trip_id', 'time_bin_index']).first().reset_index()

        # 2시간 단위 기준 속도 95% 백분위수 계산
        link_percentile95_df = unique_ps_on_candidate_links_with_timebin_gdf.groupby(['link_id']).\
            agg(percentile95=('speed', nearest_quantile_95)).reset_index()

        # 10분 단위 기준 속도 중위값 계산
        link_median_df = unique_ps_on_candidate_links_with_timebin_gdf.groupby(['link_id', 'time_bin_index']).agg(median=('speed', 'median')).reset_index()
        self.link_score_df = link_median_df.merge(link_percentile95_df, on='link_id', how='left')
        self.link_score_df['score'] = self.link_score_df['median'] / self.link_score_df['percentile95']

        return self.link_score_df
