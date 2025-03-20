import matplotlib.pyplot as plt
import sys 

from accident_matching import AccidentDataPreprocessing, AccidentMatching

def visualize_all_traj():
    pass

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
    plt.show()


if __name__ == '__main__':

    tass_data_path = 'tass_dataset/tass_2019_to_2023_241212.csv'
    ps_data_path = 'traj_sample/alltraj_20231211.txt'
    moct_network_path = 'moct_link/link'

    seoul_data = AccidentDataPreprocessing(tass_data_path, ps_data_path, moct_network_path, True)
    ps_near_acctime_in_seoul_gdf = seoul_data.link_ps_merge_sampling()

    sample = next(ps_near_acctime_in_seoul_gdf)
    accident_matching_process = AccidentMatching(radius=300)
    link_candidates = accident_matching_process.extract_link_candidate(seoul_data.tass_sample_gdf,
                                                                       seoul_data.moct_link_gdf)
    
    visualize_candidate_links(link_candidates, seoul_data.moct_link_gdf, 
                              sample, seoul_data.tass_sample_gdf)

    sys.exit(0)