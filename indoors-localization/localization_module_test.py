"""
This script showcases how to solve the localization problem making use of
the functions implemented in python classes
"""

from loc_module import EnvironmentModel, show_results_from_csv

def main():
    my_map=EnvironmentModel('Friburgo/Friburgo_Train/*.jpeg')
    my_map.import_test_images('Friburgo/Friburgo_Test_ext/*.jpeg')

    #my_map.get_map_coords()
    #my_map.export_map_coords()
    my_map.import_map_coords('map_coordinates.csv')

    #my_map.create_hog_map()
    #my_map.export_map_descriptors('HOG')
    my_map.import_map_descriptors('HOG_model.csv')

    #my_map.online_hog_test()
    #my_map.export_test_results('HOG')
    #my_map.import_test_results('batch_location_HOG.csv')

    #my_map.show_test_results()
    #show_results_from_csv('batch_location_HOG.csv')
    #show_results_from_csv('hierarchical_location_NEW.csv')

    my_map.get_cluster_labels('GIST_MATLAB_model.csv',10)
    #my_map.export_cluster_labels(3)
    #my_map.import_cluster_labels('labels3.csv')
    my_map.export_hierarchical_map_descriptors('hog10')
    #my_map.plot_clusters()

    my_map.get_representative_descriptors()
    #my_map.export_representatives()

    my_map.online_hierarchical_hog_test()
    my_map.show_test_results()
    #my_map.export_test_results()



if __name__ == "__main__":
    main()