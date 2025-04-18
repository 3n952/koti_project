import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
from tqdm import tqdm


# TODO: 기능 구현
def check_valid_dtg(candidate_links: set, input_df: gpd.GeoDataFrame):
    '''
    매칭 모듈 정확도 향상을 위해 비교적 완전한 데이터인지 검증
    args:
        trip_df: network에 올라간 ps 데이터 dataFrame
    '''
    trip_counts = input_df.groupby(['link_id', 'time_bin_index'])['trip_id'].nunique().reset_index(name='trip_count')
    # trip_count가 5 초과인 경우만 추출
    filtered_counts = trip_counts[trip_counts['trip_count'] > 5]
    # 각 link_id가 몇 개의 time_bin_index에서 조건을 만족했는지 계산
    valid_links = filtered_counts.groupby('link_id').size()

    # 사용가능 한 링크 파악
    valid_links = set(valid_links[valid_links >= 6].index)

    if candidate_links != valid_links:
        raise MatchingInvalidError()


def save_dd2taas(gt_fname, save_path):
    '''
    돌발상황 데이터를 taas 데이터화(검증 데이터 셋 구축용)
    '''
    gt_path = f'ground_truths/{gt_fname}'
    gt_df = pd.read_csv(gt_path, encoding='UTF-8')
    gt_df = gt_df[['돌발일시', '링크아이디', '거리', 'X', 'Y']]
    # 데이터 정합성 보장을 위한 사고 지점 거리 100 이하만 필터링
    gt_df = gt_df[gt_df['거리'] < 100]
    gt_df['geometry'] = gt_df.apply(lambda row: Point(row['X'], row['Y']), axis=1)
    # 좌표계 설정: UTM-K EPSG:5179 (MOCT기준)
    gdf = gpd.GeoDataFrame(gt_df, geometry='geometry', crs="EPSG:4326")
    gdf = gdf.to_crs(epsg=5179)

    # taas용 columns
    columns = ['acdnt_no', 'acdnt_year', 'acdnt_dd_dc', 'dfk_dc', 'tmzon_div_1_dc',
               'occrrnc_time_dc', 'legaldong_name', 'acdnt_hdc', 'acdnt_mdc', 'acdnt_dc',
               'lrg_violt_1_dc', 'x_crdnt_crdnt', 'y_crdnt_crdnt', 'wether_sttus_dc',
               'road_stle_dc', 'acdnt_gae_dc', 'dprs_cnt', 'sep_cnt', 'slp_cnt',
               'inj_aplcnt_cnt', 'wrngdo_vhcle_asort_hdc', 'wrngdo_vhcle_asort_dc',
               'injury_dgree_1_hdc', 'injury_dgree_1_dc', 'acdnt_age_1_dc',
               'sexdstn_div_1_dc', 'dmge_vhcle_asort_hdc', 'dmge_vhcle_asort_dc',
               'injury_dgree_2_hdc', 'injury_dgree_2_dc', 'acdnt_age_2_dc',
               'sexdstn_div_2_dc', 'rdse_sttus_dc', 'acdnt_pos', 'road_div', 'route_nm']

    # taas화
    result_df = pd.DataFrame(columns=columns)
    result_df['acdnt_dd_dc'] = gdf['돌발일시'].apply(lambda x: x[:-9])
    result_df['occrrnc_time_dc'] = gdf['돌발일시'].apply(lambda x: x[-9:-6] + '시')
    result_df['x_crdnt_crdnt'] = gdf['geometry'].apply(lambda x: x.x)
    result_df['y_crdnt_crdnt'] = gdf['geometry'].apply(lambda x: x.y)
    result_df['link_id'] = gdf['링크아이디']
    result_df = result_df[['acdnt_dd_dc', 'occrrnc_time_dc', 'x_crdnt_crdnt', 'y_crdnt_crdnt', 'link_id']]
    result_df.to_csv(save_path, encoding='euc-kr', index=False)


def save_interested_links(tass_data_path, moct_network_path, save_path):
    # tqdm 적용
    tqdm.pandas()
    # 사고 지점 로드
    taas_csv = pd.read_csv(tass_data_path, encoding='euc-kr')
    taas_csv['geometry'] = [Point(xy) for xy in zip(taas_csv['x_crdnt_crdnt'], taas_csv['y_crdnt_crdnt'])]
    taas_gdf = gpd.GeoDataFrame(taas_csv, geometry='geometry', crs='EPSG:5179')

    # 도로 링크 로드
    moct_gdf = gpd.read_file(moct_network_path, encoding='euc-kr')
    moct_sindex = moct_gdf.sindex

    # 링크 ID 저장 리스트
    all_link_ids = []

    # 각 포인트마다 100m 이내의 link_id 추출
    def extract_link_ids(point, radius=100):
        buffer = point.buffer(radius)
        idx = list(moct_sindex.intersection(buffer.bounds))
        candidates = moct_gdf.iloc[idx]
        filtered = candidates[candidates.geometry.distance(point) <= radius]
        return filtered['link_id'].dropna().unique()

    # 포인트별로 링크 ID 추출 및 누적
    for geom in tqdm(taas_gdf['geometry']):
        link_ids = extract_link_ids(geom)
        all_link_ids.extend(link_ids)  # 리스트에 하나씩 추가

    # 중복 제거 (필요한 경우만)
    all_link_ids = list(map(str, all_link_ids))  # 문자열로 변환

    # 파일 저장
    with open(save_path, 'w', encoding='utf-8') as f:
        for lid in all_link_ids:
            f.write(f"{lid}\n")

    print(f"링크 ID가 저장 완료: {save_path}")


class MatchingInvalidError(Exception):
    '''
    후보 링크 내의 DTG 데이터가 점수 산출에 유효하지 않다고 판단될 때 발생하는 에러
    '''
    def __init__(self):
        self.message = 'invalid DTG data'
        super().__init__(self.message)  # Exception 클래스 초기화

    def __str__(self):
        return f"[MatchingInvalidError] {self.message}"
