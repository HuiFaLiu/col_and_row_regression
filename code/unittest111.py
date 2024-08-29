import unittest
from unittest.mock import patch, mock_open
import numpy as np
from regression_for_5cols import *

class TestRegressionFunctions(unittest.TestCase):

    def setUp(self):
        # Mock global variables and initialize them
        global numbers_of_col, numbers_of_row, col_slope_error_threshold, max_intersection_area_ratio
        global center_xy, center_to_rectangle, rectangle_to_center, col_k_list, col_b_list, row_k_list, row_b_list
        global row_k_list_by_regression, row_b_list_by_regression, points_for_col_regression, coordinates_for_col_regression, points_for_row_regression
        global points_to_matrix_dict, points_for_row_regression_by_rect, coordinates_for_row_regression_by_rect, Matrix
        global nums_of_col, nums_of_row
        global coordinates, img_size, img_name, save_dir
        global all_img_flag, file_path, save_path, img_path, save_flag
        global fig, ax

        numbers_of_col = 5
        numbers_of_row = 6
        col_slope_error_threshold = 69.0661
        max_intersection_area_ratio = 0.01467314308844363
        center_xy = []
        center_to_rectangle = {}
        rectangle_to_center = {}
        col_k_list = []
        col_b_list = []
        row_k_list = []
        row_b_list = []
        row_k_list_by_regression = []
        row_b_list_by_regression = []
        points_for_col_regression = []
        coordinates_for_col_regression = []
        points_for_row_regression = []
        Matrix = []
        points_to_matrix_dict = {}
        points_for_row_regression_by_rect = []
        coordinates_for_row_regression_by_rect = []
        nums_of_col = 5
        nums_of_row = 6
        coordinates = []
        img_size = (1000, 1000)
        img_name = "test_image"
        save_dir = "./test_result"
        all_img_flag = False
        file_path = "./test_data/labels/vacant2_7.txt"
        save_path = "./test_result"
        img_path = file_path
        save_flag = False
        fig, ax = None, None

    @patch('builtins.open', new_callable=mock_open, read_data="10 10 20 20\n20 20 20 20\n(1000, 1000)")
    def test_read_rectangles_from_txt_happy_path(self, mock_file):
        expected_rectangles = [((0, 0), (20, 20)), ((10, 10), (30, 30))]
        expected_image_size = (1000, 1000)
        rectangles, image_size = read_rectangles_from_txt(file_path)
        self.assertEqual(rectangles, expected_rectangles)
        self.assertEqual(image_size, expected_image_size)

    @patch('builtins.open', new_callable=mock_open, read_data="10 10 20 20\n20 20 20 20\n(1000, 1000)")
    def test_read_rectangles_from_txt_invalid_image_size(self, mock_file):
        mock_file.return_value.__enter__.return_value.readlines.return_value = ["10 10 20 20\n20 20 20 20\nInvalidSize"]
        with self.assertRaises(ValueError):
            read_rectangles_from_txt(file_path)

    def test_remove_plots_from_points_happy_path(self):
        points = [(1, 2), (3, 4), (5, 6)]
        plots = [(3, 4)]
        expected_result = [(1, 2), (5, 6)]
        result = remove_plots_from_points(points, plots)
        self.assertEqual(result, expected_result)

    def test_remove_plots_from_points_no_plots_to_remove(self):
        points = [(1, 2), (3, 4), (5, 6)]
        plots = [(7, 8)]
        expected_result = [(1, 2), (3, 4), (5, 6)]
        result = remove_plots_from_points(points, plots)
        self.assertEqual(result, expected_result)

    def test_find_nearest_point_happy_path(self):
        point = (0, 0)
        points = [(1, 1), (2, 2), (3, 3)]
        expected_result = (1, 1)
        result = find_nearest_point(point, points)
        self.assertEqual(result, expected_result)

    def test_find_nearest_point_no_points(self):
        point = (0, 0)
        points = []
        expected_result = None
        result = find_nearest_point(point, points)
        self.assertEqual(result, expected_result)

    def test_find_nth_largest_y_coordinate_happy_path(self):
        points = [(1, 1), (2, 2), (3, 3)]
        n = 2
        expected_result = (2, 2)
        result = find_nth_largest_y_coordinate(points, n)
        self.assertEqual(result, expected_result)

    def test_find_nth_largest_y_coordinate_not_enough_points(self):
        points = [(1, 1), (2, 2)]
        n = 3
        expected_result = None
        result = find_nth_largest_y_coordinate(points, n)
        self.assertEqual(result, expected_result)

    def test_linear_regression_happy_path(self):
        points = np.array([[1, 1], [2, 2], [3, 3]])
        expected_k = 1.0
        expected_b = 0.0
        k, b = linear_regression(points)
        self.assertAlmostEqual(k, expected_k)
        self.assertAlmostEqual(b, expected_b)

    def test_linear_regression_single_point(self):
        points = np.array([[1, 1]])
        expected_k = 0.0
        expected_b = 1.0
        k, b = linear_regression(points)
        self.assertAlmostEqual(k, expected_k)
        self.assertAlmostEqual(b, expected_b)

    def test_is_line_intersect_rectangle_happy_path(self):
        k = 1.0
        b = 0.0
        rectangle = ((0, 0), (2, 2))
        result = is_line_intersect_rectangle(k, b, rectangle)
        self.assertTrue(result)

    def test_is_line_intersect_rectangle_no_intersection(self):
        k = 1.0
        b = 10.0
        rectangle = ((0, 0), (2, 2))
        result = is_line_intersect_rectangle(k, b, rectangle)
        self.assertFalse(result)

    def test_is_line_intersect_rectangle_vertical_line(self):
        k = 0.0
        b = 0.0
        rectangle = ((0, 0), (2, 2))
        with self.assertRaises(SystemExit):
            is_line_intersect_rectangle(k, b, rectangle)

    def test_rect_inter_ratio_happy_path(self):
        rect1 = ((0, 0), (2, 2))
        rect2 = ((1, 1), (3, 3))
        expected_result = (1, 1/7)
        result = rect_inter_ratio(rect1, rect2)
        self.assertEqual(result, expected_result)

    def test_rect_inter_ratio_no_intersection(self):
        rect1 = ((0, 0), (2, 2))
        rect2 = ((3, 3), (4, 4))
        expected_result = (0, 0.0)
        result = rect_inter_ratio(rect1, rect2)
        self.assertEqual(result, expected_result)

    def test_merge_rectangles_happy_path(self):
        rect1 = ((0, 0), (2, 2))
        rect2 = ((1, 1), (3, 3))
        max_intersection_area_ratio = 0.01
        expected_result = (1, ((0.5, 0.5), (2.5, 2.5)), 1/7)
        result = merge_rectangles(rect1, rect2, max_intersection_area_ratio)
        self.assertEqual(result[0], expected_result[0])
        self.assertEqual(result[2], expected_result[2])

    def test_merge_rectangles_no_intersection(self):
        rect1 = ((0, 0), (2, 2))
        rect2 = ((3, 3), (4, 4))
        max_intersection_area_ratio = 0.01
        expected_result = (0, None, 0.0)
        result = merge_rectangles(rect1, rect2, max_intersection_area_ratio)
        self.assertEqual(result, expected_result)

    def test_merge_rectangles_below_threshold(self):
        rect1 = ((0, 0), (2, 2))
        rect2 = ((1, 1), (3, 3))
        max_intersection_area_ratio = 0.2
        expected_result = (0, None, 0.0)
        result = merge_rectangles(rect1, rect2, max_intersection_area_ratio)
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()