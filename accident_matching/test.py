from accident_matching import AccidentMatching, AccidentDataPreprocessing


class AccidentMapMatchingProcess(AccidentDataPreprocessing, AccidentMatching):
    def __init__(self, tass_data_path, ps_data_path, moct_network_path):
        AccidentDataPreprocessing.__init__(self, tass_data_path, ps_data_path, moct_network_path)
        AccidentMatching.__init__(self)
    
    def run(self):
        self.moct_network2gdf()
        self.ps_data2gdf()
        self.tass_data2gdf()
        self.tass_sampling(ymd='20231211')
        self.merge_link_ps()
        ps_near_acctime_in_seoul_gdf = self.link_ps_merge_sampling()

        self.candidate_links_with_timebin(
            tass_sample_gdf=self.tass_sample_gdf, 
            moct_link_gdf=self.moct_link_gdf, 
            link_ps_merge_near_acctime_gdf=ps_near_acctime_in_seoul_gdf
        )

        # 사고 시간이 해당되는 날에 대한 link score
        self.candidate_link_score()

        print(link_score)


def main():
    tass_data_path = 'tass_dataset/tass_2019_to_2023_241212.csv'
    ps_data_path = 'traj_sample/taxi_20231211.txt'
    moct_network_path = 'moct_link/link'

    accidentlinkmatching = AccidentMapMatchingProcess(tass_data_path, ps_data_path, moct_network_path)
    accidentlinkmatching.run()


if __name__ == '__main__':
    main()
