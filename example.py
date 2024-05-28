from data_extractor import DataLoader, TrackerType, ResampleBy
from data_analysis import DataAnalyzer

plot_folder_name = ""
chronometer_file_name = "mock_data/chronometer.csv"

# Import chronometer data
chronometer_data = DataLoader(
    file_path=chronometer_file_name,
    tracker_type=TrackerType.CHRONOMETER,
    resample_by=ResampleBy.DAY,
)
chronometer_feature_list = chronometer_data.get_feature_list()
chrondataframe = chronometer_data.get_dataframe_by_feature(chronometer_feature_list[0])
chronometer_data.plot_single_data_vs_time(
    dataframe=chrondataframe, save_file_name=plot_folder_name + "myfile"
)

# Import keto mojo data
ketomojo_file_name = "mock_data/ketomojo.csv"
ketomojo_data = DataLoader(
    file_path=ketomojo_file_name,
    tracker_type=TrackerType.KETOMOJO,
    resample_by=ResampleBy.DAY,
    allow_multiple_samples_per_day=True
)
ketomojo_feature_list = ketomojo_data.get_feature_list()
ketodataframe0 = ketomojo_data.get_dataframe_by_feature(ketomojo_feature_list[0])
ketodataframe1 = ketomojo_data.get_dataframe_by_feature(ketomojo_feature_list[1])

ketomojo_data.plot_single_data_vs_time(
    dataframe=ketodataframe0, save_file_name=plot_folder_name + "myfile"
)

my_analyzer = DataAnalyzer(dataframes=[ketodataframe0, ketodataframe1, chrondataframe],
					       save_folder_location = plot_folder_name)
features = my_analyzer.get_feature_list()

my_analyzer.make_heatmap()
my_analyzer.plot_single_feature_vs_time(features[0])
my_analyzer.do_linear_regression(target_feature=features[0], predictor_features=[features[1]])
my_analyzer.do_linear_regression(target_feature=features[0])
my_analyzer.do_standardized_linear_regression(target_feature=features[0])
my_analyzer.make_scatter_plot(target_x=features[0], target_y=features[1])
