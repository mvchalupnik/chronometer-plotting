"""
Classes for analyzing data from Chronometer or KetoMojo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

class DataAnalyzer:
    # Set class column name constants
    analyzer_date_column_name = "Date"
    string_length_char_limit = 5

    def __init__(
        self,
        dataframes: "List[pd.DataFrame]",
        save_folder_location: str = "",
        merge_type: str = "inner",
        imputer: str = None,
    ):
        """
        Takes a list of dataframes. Saves the means of each dataframe to a new dataframe, doing imputation,
        data scaling, etc as specified.

        :param dataframes: A list of dicts (processed by the DataLoader class) which we wish to compare data from/between.
        :param save_folder_location: The location of the folder to save plots to.
        :param merge_type: Use 'inner' to merge without keeping values corresponding to indices not shared with both dataframes.
            Use 'outer' to merge while keeping values corresponding to indices not shared with both (will result in NaNs).
        :param imputer: optional choice for imputing data while importing. Options include:
            'none', 'nearest', 'ffill', 'bfill', 'quadratic', 'linear'.

        # TODO transform (log, etc.), cross-validation, train/test, ridge/lasso regression
        """
        self.save_folder_location = save_folder_location  # Folder to save plots in

        self.imputer = imputer
        self.merge_type = merge_type

        # Create and save composite dataframe, also create and save descriptive strings
        self.dataframe, self.df_descriptive_string, self.info_string = self.build_composite_dataframe(
            dataframes
        )

    def get_feature_list(self):
        """
        Return list of feature names.
        """
        return self.dataframe.columns

    def build_composite_dataframe(self, dataframes: "List[pd.DataFrame]"):
        """
        Build a new dataframe out of the average values of a list of passed-in data frames, which have
        been processed by DataLoader.
        This new dataframe can then be used for further statistical analysis in a DataAnalyzer class instance

        :param dataframes: A list of dataframes to merge.
        """

        # Create base dataframe from first dataframe in list
        first_dataframe = dataframes[0]
        base_df = pd.DataFrame(
            data={
                self.analyzer_date_column_name: first_dataframe.index.values,
                first_dataframe.name: first_dataframe.means.values,
            }
        )
        base_df.set_index(self.analyzer_date_column_name, inplace=True)

        # Create descriptive string for new dataframe
        df_descriptive_string = first_dataframe.name.split(" ")[0][:self.string_length_char_limit] + "_"

        # Loop through dataframes in the list, and add them by "means".
        for df in dataframes[1:]:
            feature_df = pd.DataFrame(
                data={
                    self.analyzer_date_column_name: df.index.values,
                    df.name: df.means.values,
                }
            )
            feature_df.set_index(self.analyzer_date_column_name, inplace=True)

            # Merge according to merge_type
            base_df = base_df.merge(feature_df, on=self.analyzer_date_column_name, how=self.merge_type, sort=True)

            # Imputation of missing values
            if self.imputer:
                base_df.interpolate(
                    method=self.imputer, inplace=True,
                )
                # Some imputation methods will still leave NaN values afterwards. If this occurs,
                # drop those rows.
                base_df.dropna(axis=0, inplace=True)

            processed_column_name = df.name.split(" ")[0][:self.string_length_char_limit] + "_"
            df_descriptive_string = df_descriptive_string + processed_column_name

        # Create descriptive info string
        info_string = (
            "imputer-"
            + str(self.imputer)
        )

        return base_df, df_descriptive_string, info_string

    def plot_single_feature_vs_time(self, feature: str):
        """
        Plot a single data from dataframe vs time.

        :param feature: The feature to plot.
        """
        fig, ax = plt.subplots()

        ax.plot(self.dataframe.index, self.dataframe[feature], "-o", color="blue")

        ax.set(xlabel="Time", ylabel=feature, title=feature)
        plt.xticks(rotation=60)
        ax.grid()
        plt.tight_layout()

        fig.savefig(
            self.save_folder_location
            + "timeseries_"
            + self.df_descriptive_string
            + self.info_string
            + ".png"
        )

    def do_linear_regression(self, target_feature:str, predictor_features: list = None):
        """
        Do linear regression on the target feature. All other features will be regressed
        against this target feature.

        :param target_feature: Target feature to perform linear regression on.
        :param predictor_features: List of features to regress against. If "None",
            all other features in the dataframe will be used.
        """
        y = self.dataframe[target_feature].values
        if predictor_features:
            X = self.dataframe[predictor_features].values
            X_labels = self.dataframe[predictor_features].columns
        else:
            X = self.dataframe.drop(target_feature, axis=1).values
            X_labels = self.dataframe.drop(target_feature, axis=1).columns

        reg = LinearRegression()
        reg.fit(X, y)

        # If single-variable X linear regression, plot.
        if X.shape[1] == 1:
            prediction_space = np.linspace(min(X), max(X)).reshape(-1, 1)

            fig = plt.figure()
            plt.title(
                "Scatter plot" + " and linear regression"
            )
            plt.ylabel(target_feature)
            plt.xlabel(predictor_features[0])

            plt.scatter(X, y, color="blue")
            plt.plot(
                prediction_space,
                reg.predict(prediction_space),
                color="black",
                linewidth=3,
            )

            # Add R^2 and standard error:
            # r^2 : metric used to quantify linear regression performance.
            # Equal to 1 - error^2/variance(y) = 1 - sum((y_pred-y)^2)/sum((y_avg - y)^2).
            # An r^2 of 1 means all of your response data fall on your model; an r^2 of 0 means
            # the model explains none of the variability of response data.
            r2 = reg.score(X, y)
            print("R^2 value: " + str(r2))

            # RMSE: Root mean squared error, another metric for linear regression.
            # Equal to sqrt(sum(y_pred - y)^2/n).
            y_pred = reg.predict(X)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            print("Root Mean Squared Error: {}".format(rmse))

            print("Coeff: {}".format(reg.coef_[0]))

            plt.text(
                0.8 * max(X),
                0.8 * max(y),
                "$R^2$: {:.3f}".format(r2)
                + "\nRMSE: {:.3f}".format(rmse)
                + "\n$y = wx + b$"
                + "\nw = {:.3f}".format(reg.coef_[0])
                + "\nb = {:.3f}".format(reg.intercept_),
            )

            plt.show()
            fig.savefig(
                self.save_folder_location
                + "linearreg-"
                + self.df_descriptive_string
                + self.info_string
                + ".png"
            )
        else:
            # If multiple variable linear regression, do not plot.
            y_pred = reg.predict(X)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            print("Root Mean Squared Error: {}".format(rmse))

            r2 = reg.score(X, y)
            print("R^2 value: " + str(r2))

            print("Coefficients: ")
            print(X_labels)
            print(reg.coef_)

    def do_standardized_linear_regression(self, target_feature:str, predictor_features: list = None):
        """
        Do a linear regression with standardized data. Print coefficients.

        :param target_feature: Target feature to perform linear regression on.
        :param predictor_features: List of features to regress against. If "None",
            all other features in the dataframe will be used.
        """
        y = self.dataframe[target_feature].values
        if predictor_features:
            X = self.dataframe[predictor_features].values
            X_labels = self.dataframe[predictor_features].columns
        else:
            X = self.dataframe.drop(target_feature, axis=1).values
            X_labels = self.dataframe.drop(target_feature, axis=1).columns

        # Scale each feature
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        reg = LinearRegression()
        reg.fit(X_scaled, y)

        # If multiple variable linear regression, do not plot.
        y_pred = reg.predict(X_scaled)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        print("Root Mean Squared Error: {}".format(rmse))

        r2 = reg.score(X_scaled, y)
        print("R^2 value: " + str(r2))

        print("Coefficients: ")
        print(X_labels)
        print(reg.coef_)

    def make_scatter_plot(self, target_x:str, target_y:str):
        """
        Plot a scatter plot for two features, using kernel density estimation to
        estimate the PDF.

        :param target_x: feature to plot along the x axis.
        :param target_y: feature to plot alon the y axis.
        """
        y = self.dataframe[target_y].values
        X = self.dataframe[target_x].values

        fig = plt.figure()

        # Kernel Density Estimate (KDE)
        values = np.vstack((X, y))
        kernel = gaussian_kde(values)
        kde = kernel.evaluate(values)

        # create array with colors for each data point
        norm = Normalize(vmin=kde.min(), vmax=kde.max())
        colors = cm.ScalarMappable(norm=norm, cmap="viridis").to_rgba(kde)

        # override original color argument
        kwargs = {}
        kwargs["color"] = colors

        plt.scatter(X, y, **kwargs)
        plt.ylabel(target_y)
        plt.xlabel(target_x)
        plt.title("Scatter plot")
        plt.show()
        fig.savefig(
            self.save_folder_location
            + "scatterplot-"
            + self.df_descriptive_string
            + self.info_string
            + ".png"
        )

    def make_heatmap(self):
        """
        Plot a heatmap for the data.
        Off-diagonals give the Pearson correlation coefficient, AKA covariance(X,Y)/(var(X)* var(Y)).
        (i, i) matrix entries are 1.
        """
        fig = plt.figure()
        plt.title("Pearson Correlation Heatmap")
        sns.heatmap(self.dataframe.corr(), square=True, cmap="RdYlGn")

        locs, labels = plt.xticks()
        plt.setp(labels, rotation=90)

        locs, labels = plt.yticks()
        ax = plt.gca()
        plt.draw()
        ax.set_yticklabels(labels, rotation=0)

        plt.tight_layout()
        plt.show()
        fig.savefig(
            self.save_folder_location
            + "heatmap-"
            + self.df_descriptive_string
            + self.info_string
            + ".png"
        )
