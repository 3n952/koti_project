import matplotlib.pyplot as plt
import sys 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from shapely.geometry import LineString, Point

from accident_matching import AccidentDataPreprocessing, AccidentMatching


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

def test_visualize(data):
    df = pd.DataFrame(data)
    # 초기 설정
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    # 초기 점 찍기
    sc = ax.scatter(df["longitude"], df["latitude"], c="red")
    ax.set_xlim(126.976, 126.985)
    ax.set_ylim(37.565, 37.575)
    ax.set_xlabel("UTM-K X")
    ax.set_ylabel("Y")
    ax.set_title("GPS 이동 경로")
    # 슬라이더 추가
    ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])
    slider = Slider(ax_slider, "Time", 0, len(df)-1, valinit=0, valstep=1)
    # 슬라이더 업데이트 함수
    def update(val):
        idx = int(slider.val)
        sc.set_offsets([[df.loc[idx, "longitude"], df.loc[idx, "latitude"]]])
        ax.set_title(f"GPS 이동 경로 - 시간: {df.loc[idx, 'timestamp']}")
        fig.canvas.draw_idle()
    slider.on_changed(update)
    plt.show(block=True)

def test2_visualize():

    # 링크별 점수 차이(최종 df)와 link geometry merge한 df 
    np.random.seed(42)
    link_ids = np.arange(1, 11)  # 10개의 링크
    link_gdf = gpd.GeoDataFrame({
        "link_id": link_ids,
        "geometry": [LineString([(126.9 + i*0.01, 37.5 + i*0.005), (126.91 + i*0.01, 37.51 + i*0.005)]) for i in range(len(link_ids))]
    })
    timestamps = pd.date_range("2024-03-20 00:00", periods=10, freq="H")  # 10시간
    score_data = []
    for t in timestamps:
        for link in link_ids:
            score_data.append([link, t, np.random.uniform(0, 1)])  # 0~1 사이 랜덤 점수
    score_df = pd.DataFrame(score_data, columns=["link_id", "timestamp", "score"])

    # 🔹 2. 초기 시각화 설정
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2)  # 슬라이더 공간 확보
    ax.set_title("시간별 링크 점수 시각화")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # 🔹 3. 색상 매핑 (점수가 낮을수록 초록, 높을수록 빨강)
    norm = mcolors.Normalize(vmin=0, vmax=1)  # 점수 범위 정규화
    cmap = cm.RdYlGn_r  # 초록-노랑-빨강 (역순)

    # 초기 데이터 플롯: timebin_idx의 첫번째 즉 -3에 대해 초기화
    time_idx = 0  # 첫 번째 시간
    filtered_scores = score_df[score_df["timestamp"] == timestamps[time_idx]]

    # 링크별 라인 추가
    lines = []
    for _, row in link_gdf.iterrows():
        link_id = row["link_id"]
        score = filtered_scores[filtered_scores["link_id"] == link_id]["score"].values[0]
        color = cmap(norm(score))

        # ✅ LineString인지 확인하고 플로팅
        if isinstance(row["geometry"], LineString):
            x, y = row["geometry"].xy
            line, = ax.plot(x, y, color=color, linewidth=3, alpha=0.8)
            lines.append(line)
        elif isinstance(row["geometry"], Point):  # 만약 Point라면 scatter로 표시
            point = row["geometry"]
            sc = ax.scatter(point.x, point.y, color=color, s=50, edgecolor='black')
            lines.append(sc)

    # 🔹 5. 슬라이더 추가
    ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])
    slider = Slider(ax_slider, "시간", 0, len(timestamps)-1, valinit=0, valstep=1)

    # 🔹 6. 슬라이더 값 변경 시 업데이트 함수
    def update(val):
        time_idx = int(slider.val)
        filtered_scores = score_df[score_df["timestamp"] == timestamps[time_idx]]

        for i, row in link_gdf.iterrows():
            link_id = row["link_id"]
            score = filtered_scores[filtered_scores["link_id"] == link_id]["score"].values[0]
            color = cmap(norm(score))

            # ✅ LineString인지 확인하고 업데이트
            if isinstance(row["geometry"], LineString):
                lines[i].set_color(color)
            elif isinstance(row["geometry"], Point):  # Point의 경우 색상 업데이트
                lines[i].set_facecolor(color)

        ax.set_title(f"시간: {timestamps[time_idx].strftime('%Y-%m-%d %H:%M')}")
        fig.canvas.draw_idle()

    # 슬라이더 이벤트 연결
    slider.on_changed(update)

    plt.show()


if __name__ == '__main__':

    #파일형식 맞춰야함
    # tass_data_path = 'tass_dataset/tass_2019_to_2023_241212.csv'
    # ps_data_path = 'traj_sample/alltraj_20231211.txt'
    # moct_network_path = 'moct_link/link'

    # seoul_data = AccidentDataPreprocessing(tass_data_path, ps_data_path, moct_network_path)
    # ps_near_acctime_in_seoul_gdf = seoul_data.link_ps_merge_sampling()

    # sample = next(ps_near_acctime_in_seoul_gdf)
    # accident_matching = AccidentMatching(radius=300)
    # accident_matching.get_candidate_links() # self.candidate_links
    
    # visualize_candidate_links(accident_matching.candidate_links, seoul_data.moct_link_gdf, 
    #                           sample, seoul_data.tass_sample_gdf)

    data = {
    "timestamp": [-3, -2, -1, 0, 1],
    "latitude": [37.5665, 37.567, 37.568, 37.569, 37.570],
    "longitude": [126.9780, 126.979, 126.980, 126.981, 126.982]
    }
    test_visualize(data)
    test2_visualize()

    sys.exit(0)