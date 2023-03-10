from predict_fed.models.decision_tree import DecisionTree as DecisionTree
from predict_fed.data import FedDecisions, FRED, Measure
from predict_fed.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np




def depth_model_results(max_depth):

    rate = FedDecisions()

    fred_data_sources = ['PAYEMS','GDPC1', 'UNRATE']

    features = {
    FRED(series): [Measure.YoY_PCT_CHANGE] for series in fred_data_sources
    }

    decision_tree = DecisionTree('squared_error', max_depth)

    pipe = Pipeline(y=rate, features=features, model=decision_tree, bootstrap=True, normalisation=True )

    performance, (X_train, X_valid, X_test, y_train, y_valid, y_test) = pipe.run()

    pred, rounded_pred = pipe.predict(X_test)

    from predict_fed.plotting import rounded_scatter

    rounded_scatter(rounded_pred, y_test)

    return performance


def test_across_depth_range(depth_range):
    #Store the nest list for efficient usability
    nested_array_of_performance = []
    #Store the Tree_Depth array
    tree_depth_array = []
    #Store the R2 Score for the Model
    r2_score_array = []
    #Store the MSE_Validation Array
    mse_validation_array = []
    #Store the MSE_Train Array
    mse_train_array = []

    for i in range(1, depth_range):
        #Handle results to avoid repeating model init / api TMR
        results_handler = depth_model_results(i)

        #Add the decision tree results to storage arrays for output grouped by result type


        nested_array_of_performance.append(results_handler)
        tree_depth_array.append(results_handler[0])
        r2_score_array.append(results_handler[1])
        mse_validation_array.append(results_handler[2])
        mse_train_array.append(results_handler[3])
        #Returns a nested arrau of decision tree results named by type with [0][i] being all result types for model with max depth i

    return [nested_array_of_performance,tree_depth_array,r2_score_array,mse_validation_array,mse_train_array]


def plot_results(y_axis_variable,x_axis_variable, filename=None , **kwargs):
    #^ Taking **kwargs as a dictionary for extra format specification

    #Converting Lists of Numbers to Arrays for use in pyplot functions
    x_points = np.array(x_axis_variable)
    y_points = np.array(y_axis_variable)
    #Creating and showing the plot
    plt.plot(x_points,y_points, **kwargs)
    plt.grid(True)
    if filename is not None:
        plt.savefig(filename)

    plt.show()


def decision_tree_results(depth_range):
    test_results = test_across_depth_range(depth_range)
    #[nested_array_of_performance,tree_depth_array,r2_score_array,mse_validation_array,mse_train_array]
    plot_results(test_results[3],test_results[1])


#decision_tree_results(6)
decision_tree_results(6)



#depth_model_results(4)
#test_across_depth_range(6)

