import matplotlib.pyplot as plt
import sys 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from shapely import wkt
from shapely.geometry import LineString, Point


from process_run import AccidentMapMatchingProcessor

def visualize_candidate_links(final_candidate_links, moct_link_gdf, 
                              link_ps_merge_near_acctime_gdf, tass_sample_gdf):

    candidate_links = set(final_candidate_links.keys())
    moct_link_gdf_filtered = moct_link_gdf[moct_link_gdf['link_id'].isin(candidate_links)]
    link_ps_merge_near_acctime_gdf_filtered = link_ps_merge_near_acctime_gdf\
        [link_ps_merge_near_acctime_gdf['link_id'].isin(candidate_links)]
    tass_sample_gdf['buffer'] = tass_sample_gdf.geometry.buffer(300)

    # 시각화
    fig, ax = plt.subplots(figsize=(10, 8))
    moct_link_gdf_filtered.plot(ax=ax, color='black', linewidth=2, alpha=0.2, label="Seoul Link")
    link_ps_merge_near_acctime_gdf_filtered.plot(ax=ax, color='green', markersize=1, alpha=0.7, label="DTG Trajectory")
    tass_sample_gdf.plot(ax=ax, color='red', markersize=10, label="accident (Point)")
    tass_sample_gdf['buffer'].plot(ax=ax, color='blue', alpha=0.3, edgecolor='blue')

    plt.legend()
    plt.title("candidate Link")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show(block=True)

def check_ground_truth(gt_path, moct_link_path, taas_sample_path):
    '''
    gt(돌발상황 정보) 데이터에 나타난 사고 지점 위치 주변 링크를 시각화하여 확인하는 함수
    '''
    gt = pd.read_csv(gt_path)
    gt['geometry'] = gpd.points_from_xy(gt['X'], gt['Y'])
    gt_gdf = gpd.GeoDataFrame(gt, geometry='geometry', crs="EPSG:4326")
    gt_gdf = gt_gdf.to_crs("EPSG:5179")

    try:
        taas_sample_gdf = gpd.read_file(taas_sample_path, encoding='euc-kr')
    except UnicodeDecodeError as E:
        print(E)
        print('try decode for UTF-8')
        taas_sample_gdf = gpd.read_file(taas_sample_path, encoding='UTF-8')
    
    taas_sample_gdf['geometry'] = taas_sample_gdf['geometry'].apply(wkt.loads)
    taas_sample_gdf = gpd.GeoDataFrame(taas_sample_gdf, geometry='geometry', crs='EPSG:5179')  # 좌표계 확인 필요
    taas_point = taas_sample_gdf['geometry'].iloc[0]
    x = taas_point.x
    y = taas_point.y

    radius1, radius2 = 300, 300
    taas_sample_gdf['buffer1'] = taas_sample_gdf.geometry.buffer(radius1)
    taas_sample_gdf['buffer2'] = taas_sample_gdf.geometry.buffer(radius2)

    candidate_links = {}
    for _, row in gt_gdf.iterrows():
        gt_point = row['geometry']
        gt_link_id = row['링크아이디']  # link_id 추출

        if gt_point.is_empty:
            continue

        # 도로(링크) 전체가 반경 내에 포함되는지 검사
        # accident point는 하나의 사고 포인트
        if gt_point.distance(taas_point) <= radius1:  
            candidate_links[gt_link_id] = gt_point  # {link_id: Point(사고지점)} 형태로 저장
            print(f'taas와 반경 {radius1}(m)이내에 있는 ground Truth의 link_id : {gt_link_id}')

    try: 
        link_gdf = gpd.read_file(moct_link_path, encoding='euc-kr')
    except:
        link_gdf = gpd.read_file(moct_link_path, encoding='UTF-8')

    link_gdf = link_gdf[['link_id', 'road_rank', 'sido_id', 'sgg_id','geometry']]
    link_gdf.set_crs(epsg=5179, inplace=True)

    link_gdf_near_taas = link_gdf[link_gdf['geometry'].distance(taas_point) <= 300]
    gt_near_tass_link_id = set(candidate_links.keys()) # key 자료형 int
    gt_near_tass_gdf = link_gdf[link_gdf['link_id'].astype(int).isin(gt_near_tass_link_id)]

    # 시각화
    fig, ax = plt.subplots(figsize=(10, 8))
    link_gdf_near_taas.plot(ax=ax, color='black', linewidth=2, alpha=0.3, label=f"Candidate Link near taas({radius2}m)")
    taas_sample_gdf['buffer2'].plot(ax=ax, facecolor='blue', edgecolor='red', alpha=0.3, label=f'{radius2}')

    gt_near_tass_gdf.plot(ax=ax, color='green', linewidth=1, label=f"GT Link near taas({radius1}m)")
    taas_sample_gdf['buffer1'].plot(ax=ax, facecolor='blue', edgecolor='blue', alpha=0.3, label=f'{radius1}')
    ax.plot(x, y, 'ro', markersize=5, label="taas Point")
    

    plt.legend()
    plt.title("ground_truth vs candidate links")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show(block=True)

def visualize_matching_links(taas_sample_path, moct_network_path, answer_link_id: str):
 
    try:
        tass_sample_gdf = gpd.read_file(taas_sample_path, encoding='euc-kr')
    except UnicodeDecodeError as E:
        print(E)
        print('try decode for UTF-8')
        tass_sample_gdf = gpd.read_file(taas_sample_path, encoding='UTF-8')

    tass_sample_gdf['geometry'] = tass_sample_gdf['geometry'].apply(wkt.loads)
    tass_sample_gdf = gpd.GeoDataFrame(tass_sample_gdf, geometry='geometry', crs='EPSG:5179')  # 좌표계 확인 필요
    tass_sample_gdf['buffer'] = tass_sample_gdf.geometry.buffer(300)
    buffer_union = tass_sample_gdf['buffer'].union_all()

    result = pd.read_csv('result/result_link_score.csv')
    result = result.sort_values(by="result_score", ascending=False)
    result['link_id'] = result['link_id'].astype(int).astype(str)
    
    result_links = list()
    for i in range(10):
            result_link = result.iloc[i]
            result_links.append(result_link['link_id']) # link_id: str

    if answer_link_id in result_links:
        print('사고링크맵매칭 top 10 중 정답이 있습니다.')
    else:
        print('사고링크맵매칭 top 10 중 정답이 없습니다.')

    # moct
    try: 
        moct_network_link_gdf = gpd.read_file(moct_network_path, encoding='euc-kr')
    except:
        moct_network_link_gdf = gpd.read_file(moct_network_path, encoding='UTF-8')

    moct_network_link_gdf_og = moct_network_link_gdf[['link_id', 'road_rank', 'sido_id', 'sgg_id','geometry']]
    moct_network_link_gdf_og.set_crs(epsg=5179, inplace=True)
    moct_network_link_gdf = moct_network_link_gdf_og[moct_network_link_gdf_og.geometry.intersects(buffer_union)]

    moct_link_gdf_filtered = moct_network_link_gdf_og[moct_network_link_gdf_og['link_id'].isin(result_links)]
    moct_link_gdf_filtered = moct_link_gdf_filtered.merge(result, on="link_id",  how="inner")
    # 높은순 재정렬
    moct_link_gdf_filtered = moct_link_gdf_filtered.sort_values("result_score", ascending=False).reset_index(drop=True)

    moct_link_gdf_answered = moct_network_link_gdf_og[moct_network_link_gdf_og['link_id'].isin([answer_link_id])]

    # 시각화
    fig, ax = plt.subplots(figsize=(15, 10))
    # Taas 포인트 
    tass_sample_gdf.plot(ax=ax, color='red', markersize=20, label="Taas Point")
    # 정답 링크 그리기
    moct_link_gdf_answered.plot(ax=ax, color='red', linewidth=3, label="Answer Links")

    # 기존 moctlink in buffer 그리기 
    moct_network_link_gdf.plot(ax=ax, color='black', linewidth=2, alpha=0.3, label="MOCT Links")
    # Top 10 링크는 파란색으로 그리기
    moct_link_gdf_filtered.plot(ax=ax, color='blue', linewidth=2, alpha=0.7, label="Top 10 Links")


    # 지도에 숫자 1~10만 표시
    for idx, row in moct_link_gdf_filtered.iterrows():
        center = row['geometry'].interpolate(0.5, normalized=True)
        ax.text(center.x, center.y, str(idx + 1),
                fontsize=10, fontweight='bold', color='black',
                ha='center', va='center')

    # Top 10 link_id 목록 (범례 느낌으로 텍스트 상자에 표시)
    top10_labels = [f"Top {idx+1}: {row['link_id']}" for idx, row in moct_link_gdf_filtered.iterrows()]
    label_text = "\n".join(top10_labels)

    # 텍스트 박스 추가 (우측 상단)
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(1.02, 0.95, label_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    ax.set_aspect('auto')  # 1:1 비율 유지
    # 제목 & 범례
    ax.set_title("Top 10 Candidate Links", fontsize=12)
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.show(block=True)

def visualize_score_per_timebin(taas_sample_path, timebin_score_path, moct_network_path):

    try:
        taas_sample_gdf = gpd.read_file(taas_sample_path, encoding='euc-kr')
    except:
        taas_sample_gdf = gpd.read_file(taas_sample_path, encoding='UTF-8')

    #print(tass_sample_gdf.head())
    #tass_sample_gdf = tass_sample_gdf[['geometry']]
    taas_sample_gdf['geometry'] = taas_sample_gdf['geometry'].apply(wkt.loads)
    taas_sample_gdf = gpd.GeoDataFrame(taas_sample_gdf, geometry='geometry', crs='EPSG:5179')  # 좌표계 확인 필요
    taas_sample_gdf['buffer'] = taas_sample_gdf.geometry.buffer(300)
    buffer_union = taas_sample_gdf['buffer'].union_all()

    timebin_score_df = pd.read_csv(timebin_score_path)
    timebin_score_df.set_index('link_id', inplace=True)
    timebin_score_df = timebin_score_df.reset_index()
    timebin_score_df['link_id'] = timebin_score_df['link_id'].astype(str)
    
    timebin_columns = [col for col in timebin_score_df.columns if col.startswith("diff_")]

    try:
        moct_network_link_gdf = gpd.read_file(moct_network_path, encoding='euc-kr')
    except:
        moct_network_link_gdf = gpd.read_file(moct_network_path, encoding='UTF-8')

    moct_network_link_gdf = moct_network_link_gdf[['link_id', 'road_rank', 'sido_id', 'sgg_id', 'geometry']]
    moct_network_link_gdf.set_crs(epsg=5179, inplace=True)
    moct_network_link_gdf['link_id'] = moct_network_link_gdf['link_id'].astype(str)
    moct_network_link_gdf = moct_network_link_gdf[moct_network_link_gdf.geometry.intersects(buffer_union)]

    merged_gdf = moct_network_link_gdf.merge(timebin_score_df, on="link_id", how="inner")

    # 초기 설정
    initial_timebin_idx = 0
    initial_col = timebin_columns[initial_timebin_idx]
    values = merged_gdf[initial_col]

    # 시각화 설정
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.subplots_adjust(bottom=0.2)

    norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
    cmap = cm.RdYlGn_r
    colors = cmap(norm(values))

    # GeoDataFrame plot 초기화
    # GeoDataFrame plot 초기화
    moct_network_link_gdf.plot(ax=ax, color='black', linewidth=2, alpha=0.3, label="MOCT Links (All)")

    plot = merged_gdf.plot(ax=ax, color=colors, linewidth=2, label="Scored Links")
    ax.set_title(f"Timebin: {initial_col}")
    ax.axis('off')

    # 슬라이더 추가
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, "Timebin", 0, len(timebin_columns) - 1, valinit=initial_timebin_idx, valstep=1)

    # 슬라이더 콜백
    def update(val):
        idx = int(slider.val)
        col = timebin_columns[idx]
        new_values = merged_gdf[col]
        norm = mcolors.Normalize(vmin=new_values.min(), vmax=new_values.max())
        new_colors = cmap(norm(new_values))

        # 지도 다시 그리기
        ax.clear()
        #배경 도로: 전체 MOCT 도로망 (연하게)
        moct_network_link_gdf.plot(ax=ax, color='black', linewidth=2, alpha=0.3, label="MOCT Links (All)")
        #슬라이더로 선택된 score 기반 도로만 강조
        merged_gdf.plot(ax=ax, color=new_colors, linewidth=2, label="Scored Links")
        ax.set_title(f"Timebin: {col}")
        ax.axis('off')
        fig.canvas.draw_idle()
    slider.on_changed(update)
    plt.show(block=True)

if __name__ == '__main__':

    tass_data_path = 'tass_dataset/tass_2019_to_2023_241212.csv'
    ps_data_path = 'traj_sample/alltraj_20231211.txt'
    moct_network_path = 'moct_link/link'
    taas_sample_path = 'result/taas_sample.csv'
    gt_data_path = 'ground_truths/gt_20231211.csv'
    timebin_score_path = 'result/delta_score.csv'
    result_link_path = 'result/result_link_score.csv'
    answer_link_id = '1950442300' # 마무리한 것: 신촌, 능동, 안암, 송파 농서로

    check_ground_truth(gt_data_path, moct_network_path, taas_sample_path)
    #visualize_score_per_timebin(taas_sample_path,timebin_score_path, moct_network_path)
    #visualize_matching_links(taas_sample_path, moct_network_path, answer_link_id)

    sys.exit(0)

