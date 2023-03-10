from predict_fed.models.decision_tree import DecisionTree as DecisionTree
from predict_fed.data import FedDecisions, FRED, Measure
from predict_fed.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np


def decision_tree_max_depth(max_depth):
    rate = FedDecisions()

    fred_data_sources = ['PAYEMS', 'GDPC1', 'UNRATE']

    features = {
        FRED(series): [Measure.YoY_PCT_CHANGE] for series in fred_data_sources
    }

    decision_tree = DecisionTree('squared_error', max_depth)

    pipe = Pipeline(y=rate, features=features, model=decision_tree, bootstrap=True, normalisation=True)

    performance, (X_train, X_valid, X_test, y_train, y_valid, y_test) = pipe.run()

    pred, rounded_pred = pipe.predict(X_test)

    from predict_fed.plotting import rounded_scatter

    rounded_scatter(rounded_pred, y_test)

    return performance


def test_across_depth_range(depth_range):
    # Store the nest list for efficient usability
    nested_array_of_performance = []
    # Store the Tree_Depth array
    tree_depth_array = []
    # Store the R2 Score for the Model
    r2_score_array = []
    # Store the MSE_Validation Array
    mse_validation_array = []
    # Store the MSE_Train Array
    mse_train_array = []

    for i in range(1, depth_range):
        # Handle results to avoid repeating model init / api TMR
        results_handler = decision_tree_max_depth(i)

        # Add the decision tree results to storage arrays for output grouped by result type

        nested_array_of_performance.append(results_handler)
        tree_depth_array.append(results_handler[0])
        r2_score_array.append(results_handler[1])
        mse_validation_array.append(results_handler[2])
        mse_train_array.append(results_handler[3])
        # Returns a nested arrau of decision tree results named by type with [0][i] being all result types for model
        # with max depth i

    return [nested_array_of_performance, tree_depth_array, r2_score_array, mse_validation_array, mse_train_array]


def plot_results(y_axis_variable, x_axis_variable, filename=None, **kwargs):
    # ^ Taking **kwargs as a dictionary for extra format specification

    # Converting Lists of Numbers to Arrays for use in pyplot functions
    x_points = np.array(x_axis_variable)
    y_points = np.array(y_axis_variable)
    # Creating and showing the plot
    plt.plot(x_points, y_points, **kwargs)
    plt.grid(True)
    if filename is not None:
        plt.savefig(filename)

    plt.show()


##NEED TO EDIT BELOW FUNCTION TO USE OTHER PLOTTING FUNCS ETC
def decision_tree_results(depth_range):
    test_results = test_across_depth_range(depth_range)
    # [nested_array_of_performance,tree_depth_array,r2_score_array,mse_validation_array,mse_train_array]
    plot_results(test_results[3], test_results[1])

def decision_tree_max_depth_cross_validation_with_api_call(max_depth, number_of_chunks, chunk_number):

    rate = FedDecisions()

    fred_data_sources = ['PAYEMS','GDPC1', 'UNRATE']

    features = {
    FRED(series): [Measure.YoY_PCT_CHANGE] for series in fred_data_sources
    }

    decision_tree = DecisionTree('squared_error', max_depth)

    pipe = Pipeline(y=rate, features=features, model=decision_tree,balance=True, bootstrap=True, normalisation=True, cross_valid=True,n_chunks=number_of_chunks, chunk_n=chunk_number)

    performance, (X_train, X_valid, X_test, y_train, y_valid, y_test) = pipe.run()

    pred, rounded_pred = pipe.predict(X_test)

    from predict_fed.plotting import rounded_scatter

    rounded_scatter(rounded_pred, y_test)

    return performance


#Splitting the decision tree function to avoid timeout api
def instantiate_data_set_with_api(data_source_list=['PAYEMS','GDPC1', 'UNRATE']):
    rate = FedDecisions()

    fred_data_sources = data_source_list

    features = {
    FRED(series): [Measure.YoY_PCT_CHANGE] for series in fred_data_sources
    }
    return rate,features

def decision_tree_max_depth_cross_validation(rate, features,max_depth, number_of_chunks, chunk_number):


    decision_tree = DecisionTree('squared_error', max_depth)

    pipe = Pipeline(y=rate, features=features, model=decision_tree, bootstrap=True, normalisation=True, cross_valid=True,n_chunks=number_of_chunks, chunk_n=chunk_number)

    performance, (X_train, X_valid, X_test, y_train, y_valid, y_test) = pipe.run()

    pred, rounded_pred = pipe.predict(X_test)

    from predict_fed.plotting import rounded_scatter

    rounded_scatter(rounded_pred, y_test)

    return performance

def run_test_depth_range_cross_valid(rate,features,depth_range,number_of_chunks):
    # Returns nested lists


    #Nested lists of performance metrics of the following format:
    #metric_name[i][j] = performance metric values for chunk j of the cross validation datasplit with max tree depth of i
        # where j ranges from 0:n-1 where n =  number_of_chunks (cross validation input)
        # where i ranges from 0:m-1 where m = tree depth range

    # Performance array : [tree_depth, r2_value, validation_mse, training_mse]



    tree_depth_nested = []
    r2_value_nested = []
    validation_mse_nested = []
    training_mse_nested = []

    for i in range(1,depth_range+1):
        tree_depth_nested_depth_handler =[]
        r2_value_nested_depth_handler = []
        validation_mse_nested_depth_handler = []
        training_mse_nested_depth_handler = []
        for j in range(number_of_chunks):
            performance_results_handler = decision_tree_max_depth_cross_validation(rate,features,i, number_of_chunks,j)
            tree_depth_nested_depth_handler.append(performance_results_handler[0])
            r2_value_nested_depth_handler.append(performance_results_handler[1])
            validation_mse_nested_depth_handler.append(performance_results_handler[2])
            training_mse_nested_depth_handler.append(performance_results_handler[3])

        tree_depth_nested.append(tree_depth_nested_depth_handler)
        r2_value_nested.append(r2_value_nested_depth_handler)
        validation_mse_nested.append(validation_mse_nested_depth_handler)
        training_mse_nested.append(training_mse_nested_depth_handler)

    print(tree_depth_nested)
    return tree_depth_nested, r2_value_nested, validation_mse_nested, training_mse_nested


def nested_list_data_processing(nested_list):
    # Nested lists of performance metrics of the following format:
    # metric_name[i][j] = performance metric values for chunk j of the cross validation datasplit with max tree depth of i
    # where j ranges from 0:n-1 where n =  number_of_chunks (cross validation input)
    # where i ranges from 0:m-1 where m = tree depth range
    mean_per_depth = []
    variance_per_depth = []
    max_per_depth = []
    min_per_depth = []

    for i in nested_list:
        #i = array of metric scores across different cross valid chunks for depth i
        mean = np.mean(i)
        variance = np.var(i)
        max_value = np.amax(i)
        min_value = np.amin(i)

        mean_per_depth.append(mean)
        variance_per_depth.append(variance)
        max_per_depth.append(max_value)
        min_per_depth.append(min_value)

    return mean_per_depth,variance_per_depth,max_per_depth,min_per_depth

def test_w_print_stats(depth_range,number_of_chunks):
    rate,features = instantiate_data_set_with_api()

    tree_depth_nested, r2_value_nested, validation_mse_nested, training_mse_nested = run_test_depth_range_cross_valid(rate,features,depth_range, number_of_chunks)
    list_of_metrics = [tree_depth_nested, r2_value_nested, validation_mse_nested, training_mse_nested]
    for metric in list_of_metrics:
        mean_per_depth,variance_per_depth,max_per_depth,min_per_depth = nested_list_data_processing(metric)
        if list_of_metrics.index(metric) == 0:
            print("Tree Depth Statistics : ", "Mean : ", mean_per_depth, " Variance:", variance_per_depth, "Max Value: ", max_per_depth, " Min Value: ", min_per_depth)
        elif list_of_metrics.index(metric) == 1:
            print("R2 Value Statistics : ", "Mean : ", mean_per_depth, " Variance:", variance_per_depth, "Max Value: ", max_per_depth, " Min Value: ", min_per_depth)
        elif list_of_metrics.index(metric) == 2:
            print("Validation MSE Statistics : ", "Mean : ", mean_per_depth, " Variance:", variance_per_depth, "Max Value: ", max_per_depth, " Min Value: ", min_per_depth)

        elif list_of_metrics.index(metric) == 3:
            print("Training MSE Statistics : ", "Mean : ", mean_per_depth, " Variance:", variance_per_depth, "Max Value: ", max_per_depth, " Min Value: ", min_per_depth)

    return list_of_metrics

#test_results = test_w_print_stats(13,5)


def visualise_results(depth_range,number_of_chunks):
    rate,features = instantiate_data_set_with_api()

    tree_depth_nested, r2_value_nested, validation_mse_nested, training_mse_nested = run_test_depth_range_cross_valid(rate, features, depth_range, number_of_chunks)
    # Nested lists of performance metrics of the following format:
    # metric_name[i][j] = performance metric values for chunk j of the cross validation datasplit with max tree depth of i
    # where j ranges from 0:n-1 where n =  number_of_chunks (cross validation input)
    # where i ranges from 0:m-1 where m = tree depth range

    mean_per_depth, variance_per_depth, max_per_depth, min_per_depth = nested_list_data_processing(tree_depth_nested)
    x = np.linspace(1, len(tree_depth_nested), len(tree_depth_nested))
    plot_results(mean_per_depth, x, 'tree_depth_nested.png')

    mean_per_depth, variance_per_depth, max_per_depth, min_per_depth = nested_list_data_processing(r2_value_nested)
    x = np.linspace(1, len(r2_value_nested), len(r2_value_nested))
    plot_results(mean_per_depth, x, 'r2_value_mean.png')

    mean_per_depth, variance_per_depth, max_per_depth, min_per_depth = nested_list_data_processing(validation_mse_nested)
    x = np.linspace(1, len(validation_mse_nested), len(validation_mse_nested))
    plot_results(mean_per_depth, x, 'validation_mse.png')

    mean_per_depth, variance_per_depth, max_per_depth, min_per_depth = nested_list_data_processing(training_mse_nested)
    x = np.linspace(1, len(training_mse_nested), len(training_mse_nested))
    plot_results(mean_per_depth, x, 'training_mse.png')




visualise_results(20,50)


























