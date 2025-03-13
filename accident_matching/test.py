from accident_matching import AccidentMatching, AccidentDataPreprocessing

if __name__ == '__main__':

    tass_data_path = 'tass_dataset/tass_2019_to_2023_241212.csv'
    ps_data_path = 'traj_sample/taxi_20231211.txt'
    moct_network_path = r'D:\traffic_metric_backup\network\LINK_2024'

    seoul_data = AccidentDataPreprocessing(tass_data_path, ps_data_path, moct_network_path)

    seoul_link_gdf = seoul_data.moct_network2gdf
    korea_ps_gdf = seoul_data.ps_data2gdf
    seoul_tass_gdf = seoul_data.tass_data2gdf

    seoul_tass_sample_gdf = seoul_data.tass_sampling()
    seoul_ps_gdf = seoul_data.merge_link_ps()
    ps_near_acctime_in_seoul_gdf = seoul_data.link_ps_merge_sampling()

    accident_matching_process = AccidentMatching()
    link_candidate = accident_matching_process.extract_link_candidate(seoul_tass_sample_gdf, seoul_link_gdf)
    link_score = accident_matching_process.candidate_link_score()

    print(link_candidate)
