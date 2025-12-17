**hyperplane.lp**: the file containing the optimization problem for creating a separating hyperplane \
**hypersphere.lp**: the file containing the optimization problem for creating a separating hypersphere \
**hyperplane_solutions.csv**: csv containing the optimal solution for the separating hyperplane \
**hypersphere_solutions.csv**: csv containing optimal solution for the separating hypersphere \
**hyperplane_results.csv**: results for predictions made on test data with optimal solution for separating hyperplane \
**hypersphere_results.csv**: results for predictions made on test data with optimal solution for separating hypersphere \
**add_features.py**: file that creates dataframe for emails and adds relevant datapoints to the dataframe \
**prepare_data.py**: file that splits data into training and testing sets, creates files for the hyperplane and hypersphere, uses gurobi to solve each equation, and makes predictions on the test set \
**spam_guesser.py**: file for website that uses hyperplane equation to classify new emails 
