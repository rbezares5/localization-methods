"""
This script showcases how to solve the localization problem making use of
the functions implemented in python classes
"""

from loc_module import EnvironmentModel

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
    my_map.import_test_results('batch_location_HOG.csv')

    my_map.show_neighbours_histogram()



if __name__ == "__main__":
    main()