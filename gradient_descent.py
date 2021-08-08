"""
Gradient descent for solving the following hypothesis:

h_theta(x) = theta_0 + theta_1 * x

where cost function is:

J_theta(theta_0, theta_1) = 1/2m Σ (h_theta(x_i) - y_i)^2

Goal is to minimize J_theta (sum squared errors) and find optimal values theta_0 and theta_1
"""
import pandas as pd


class GradientDescent:
    def __init__(self, data_df):
        """ Run gradient descent algorithm on input data
            :param data_df: input data in form of a pandas dataframe with two columns x and y
        """
        theta_0, theta_1 = self.gradient_descent(data_df)
        print(f'Found optimal values for theta_0: {theta_0} and theta_1: {theta_1}')

    def gradient_descent(self, data_df):
        """
            :param data_df: input data in form of a pandas dataframe with two columns x and y
            :return teta_0: final estimated teta_0 value
            :return teta_1: final estimated teta_1 value
        """
        theta_0 = 0
        theta_1 = 0
        learning_rate = 0.1
        sample_size = data_df.shape[0]
        total_iterations = 1000
        iter_count = 0
        
        while iter_count < total_iterations:
            # Add h_theta(x_i) and h_theta(x_i) - y_i columns to the original data based on current theta values
            new_data_df = data_df.copy()
            new_data_df['h_theta'] = new_data_df.apply(lambda this_row: self.hypothesis_function(theta_0, theta_1, this_row['x']), axis = 1)
            new_data_df['h_theta_y_diff'] = new_data_df.apply(lambda this_row: this_row['h_theta'] - this_row['y'], axis = 1)
            new_data_df['h_theta_y_diff_times_x'] = new_data_df.apply(lambda this_row: this_row['h_theta_y_diff'] * this_row['x'], axis = 1)

            # Get updated theta_0 value
            # Based on equations new theta_0 value is:
            # theta_0 := theta_0 - learning_rate * 1/sample_size * Σ(h_theta(x_i) - y_i)
            new_theta_0 = theta_0 - learning_rate * (1.0 / sample_size) * new_data_df['h_theta_y_diff'].sum()

            # Get updated theta_1 value
            # Based on equations new theta_1 value is:
            # theta_1 := theta_1 - learning_rate * 1/sample_size * Σ(h_theta(x_i) - y_i) * x_i
            new_theta_1 = theta_1 - learning_rate * (1.0 / sample_size) * new_data_df['h_theta_y_diff_times_x'].sum()
            
            # If new and previous values of both thetas are the same, break the loop and return theta values
            # continue otherwise
            if new_theta_0 == theta_0 and new_theta_1 == theta_1:
                break

            # Update theta values
            theta_0 = new_theta_0
            theta_1 = new_theta_1

            # Update iter_count
            iter_count += 1

        return round(theta_0, 2), round(theta_1, 2)

    @staticmethod
    def hypothesis_function(theta_0, theta_1, x):
        """ Returns h_theta(x_i) using the following equation:
            h_theta(x_i) = theta_0 + theta_1 * x_i
            :param theta_0: current teta_0 value
            :param theta_1: current teta_1 value
            :param x: theta_0 + theta_1 * x
            :return:
        """
        return theta_0 + (theta_1 * x)


# Testing the code (code should print out teta_0 = 0 and teta_1 = 1 for this sample dataset)
# Sample data
column_names = ['x', 'y']
data_tuples = [(0.0, 0.0),
               (1.0, 1.0),
               (2.0, 2.0),
               (3.0, 3.0)]
data_df = pd.DataFrame([this_tuple for this_tuple in data_tuples], columns = column_names)
new_gradient_descent = GradientDescent(data_df = data_df)
