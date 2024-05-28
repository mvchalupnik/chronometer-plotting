"""
Classes for extracting data from Chronometer and/or KetoMojo.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from enum import Enum, auto


class TrackerType(Enum):
    CHRONOMETER = auto()
    KETOMOJO = auto()


class ResampleBy(Enum):
    DAY = "D"
    WEEK = "W"
    MONTH = "M"


class DataLoader:
    # Set class column name constants
    value_column_name = "value"
    iso_date_column_name = "iso_date"
    reformatted_iso_date_column_name = "reformatted_iso_date"
    means_column_name = "means"

    chronometer_date_column_name = "Date"
    ketomojo_date_column_name = "date"
    ketomojo_type_column_name = "type"
    ketomojo_unit_column_name = "unit"

    def __init__(
        self,
        file_path: str,
        tracker_type: TrackerType,
        resample_by: ResampleBy = ResampleBy.DAY,
        n_rows: int = None,
        start_at: "Datetime" = datetime.datetime(2023, 1, 1),
        allow_multiple_samples_per_day: bool = False,
    ):
        """
        Create a DataLoader object which loads a CSV file indicated by a string containing a file path.

        :param file_path: The location of the file containing the CSV with the tracker.
        :param tracker_type: An Enum indicating whether Chronometer or KetoMojo data is being extracted.
        :param resample_by: An Enum indicating whether to resample the data by day, week, month, or year.
        :param n_rows: The number of rows to include, starting from the latest data. If "None", include
            all rows.
        :param start_at: a date object that gives the earliest date to start including data from.
        :param allow_multiple_samples_per_day: Whether the dataframe can include.
            multiple samples per day. Affects the way standard error is calculated.
        """
        # Load CSV into dataframe
        self.dataframe = pd.read_csv(file_path)

        self.tracker_type = tracker_type
        self.resample_by = resample_by
        self.n_rows = n_rows
        self.start_at = start_at
        self.allow_multiple_samples_per_day = allow_multiple_samples_per_day

        # Set resample label
        if self.resample_by == ResampleBy.DAY:
            self.label_by = "left"
        else:
            self.label_by = "right"

        # Create a dict to store the processed dataframes
        self.processed_dataframes = {}

        # Load pandas dataframe from CSV and store; and track daily min and max
        self.feature_list = self.extract_data()

    def get_feature_list(self):
        """Return a list of features."""
        return self.feature_list

    def get_dataframe_by_feature(self, feature_name: str):
        """
        Get a dataframe by feature name.

        :param feature_name: The feature to grab the dataframe of.
        """
        return self.processed_dataframes[feature_name]

    def get_descriptive_string(self, feature_name: str):
        """
        Return a string containing information about the tracker and associated data.

        :param feature_name: The feature which we wish to describe.
        """

        # Convert n_rows to string
        if self.n_rows == 0:
            n_rows_string = "all"
        else:
            n_rows_string = str(self.n_rows)
        # convert start_at to string
        start_at_string = self.start_at.strftime("%Y-%m-%d")

        # Convert feature name to string
        feature_name_string = feature_name.split(" ")[0]

        descriptive_string = (
            self.tracker_type.name
            + "_"
            + feature_name_string
            + "_resampleby-"
            + self.resample_by.name
            + "_nrows-"
            + n_rows_string
            + "_startat-"
            + start_at_string
        )
        return descriptive_string

    def extract_data(self):
        """Load CSV file and process dataframes."""
        df = self.dataframe.copy()
        print(f"Extracting columns: {df.columns[1:]}")

        if self.tracker_type == TrackerType.CHRONOMETER:
            feature_list = df.columns[1:]
            feature_list_with_units = feature_list
        elif self.tracker_type == TrackerType.KETOMOJO:
            feature_list = df[self.ketomojo_type_column_name].unique()
            feature_list_with_units = []
        else:
            raise ValueError(
                f"{self.tracker_type} does not exist, please use a tracker type which has been implemented."
            )

        for feature in feature_list:
            if self.tracker_type == TrackerType.CHRONOMETER:
                # drop unneeded columns and rename "Date" column
                df_by_tracker = df[[self.chronometer_date_column_name, feature]].rename(
                    columns={
                        self.chronometer_date_column_name: self.iso_date_column_name,
                        feature: self.value_column_name,
                    }
                )
                feature_with_units = feature

            elif self.tracker_type == TrackerType.KETOMOJO:
                # Grab just the tracker data specified
                grouped_df = df.groupby(self.ketomojo_type_column_name).get_group(feature)

                # Drop all unnecessary columns
                df_by_tracker = grouped_df[[self.ketomojo_date_column_name, self.value_column_name]].rename(
                    columns={
                        self.ketomojo_date_column_name: self.iso_date_column_name,
                    }
                )
                # Add unit to feature name
                feature_unit_name = grouped_df[self.ketomojo_unit_column_name].unique()
                if len(feature_unit_name) != 1:
                    # TODO Convert all readings to match units (mmol/L, etc), in case they do not match
                    # For now, raise error in case they do not match
                    raise ValueError("All units must be the same for each feature.")
                feature_with_units = f"{feature} ({feature_unit_name[0]})"
                feature_list_with_units.append(feature_with_units)

                # TODO keep track of time of day
                # For now, ignore

            else:
                raise ValueError(
                    f"{self.tracker_type} does not exist, please use a tracker type which has been implemented."
                )

            # Extract data by feature
            processed_df = self.extract_feature(df_by_tracker)

            # Name the dataframe according to the feature (column) name
            processed_df.name = feature_with_units

            # Store the processed dataframe
            self.processed_dataframes[feature_with_units] = processed_df

        # Return list of features with units
        return feature_list_with_units

    def extract_feature(self, df: pd.DataFrame):
        """
        Process a feature of data in the dataframe, returning relevant summarizing statistics.

        :param df: formatted extracted pandas dataframe of the data.
        """
        # For each column, resample and track relevant statistics during resampling
        (
            df_with_updated_date,
            means,
            stdevs,
            stderrs,
        ) = self.resample_column(df, "%Y-%m-%d")

        # Resample to find maxes and mins
        resampled_maxes = df_with_updated_date.resample(
            self.resample_by.value,
            on=self.reformatted_iso_date_column_name,
            label=self.label_by,
        ).max()
        resampled_mins = df_with_updated_date.resample(
            self.resample_by.value,
            on=self.reformatted_iso_date_column_name,
            label=self.label_by,
        ).min()

        # Make a new dataframe
        resampled_df = (
            means.copy()
            .assign(
                means=means,
                stdevs=stdevs,
                stderrs=stderrs,
                resampled_maxes=resampled_maxes.value,
                resampled_mins=resampled_mins.value,
            )
            .drop(self.value_column_name, axis=1)
        )

        # Start at starting date passed in
        resampled_df = resampled_df.loc[resampled_df.index > self.start_at]

        # Only include the latest n rows in the dataframe
        if self.n_rows:
            resampled_df = resampled_df.head(self.n_rows)

        # drop na values in mean
        resampled_df = resampled_df.dropna(axis=0, subset=[self.means_column_name])

        # If plotting by month, change date to 1 (from end of month date) for plot clarity
        if self.resample_by == ResampleBy.MONTH:
            d = resampled_df.index.map(lambda s: pd.to_datetime(s).replace(day=1))
            resampled_df.index = d

        return resampled_df

    def resample_column(self, df: pd.DataFrame, date_format_string: str):
        """
        Process a column of data in the dataframe and resample by time, returning relevant summarizing statistics.

        :param df: formatted extracted pandas dataframe of the data.
        :param date_format_string: the format string to format the date to.
        """

        # Modify CSVs string format so date matches the datetime format; drop extraneous columns
        dates = df[self.iso_date_column_name].apply(
            lambda s: datetime.datetime.strptime(s, date_format_string)
        )
        df = df.assign(reformatted_iso_date=dates)
        df = df.drop(columns=self.iso_date_column_name)

        # resample by desired time span. Find means, total number in sample, and standard deviations
        means = df.resample(
            self.resample_by.value,
            on=self.reformatted_iso_date_column_name,
            label=self.label_by,
        ).mean()
        counts = df.resample(
            self.resample_by.value,
            on=self.reformatted_iso_date_column_name,
            label=self.label_by,
        ).count()
        stdevs = df.resample(
            self.resample_by.value,
            on=self.reformatted_iso_date_column_name,
            label=self.label_by,
        ).std(ddof=0)

        if self.resample_by == ResampleBy.MONTH:
            # Resampling by month lists data by the last day of each month
            max_data_counts = means.index.day
        elif self.resample_by == ResampleBy.WEEK:
            max_data_counts = 7
        elif self.resample_by == ResampleBy.DAY:
            max_data_counts = (
                2  # recover case where error simply is standard deviation
            )
        else:
            raise ValueError(
                f"{self.resample_by} does not exist, please use a ResampleBy type which has been implemented."
            )

        # Here, we calculate standard error. Standard error becomes an applicable metric in case data
        # is not tracked every day, and/or in case it is possible for multiple data points to be
        # tracked in one day.
        if self.allow_multiple_samples_per_day:
            # When multiple data points can be tracked in a day, the resampled average
            # does not necessarily represent the true population average.
            # In this case, standard error equals standard deviation over root(n)
            stderrs = stdevs[self.value_column_name] / np.sqrt(
                counts[self.value_column_name]
            )

        else:
            # When only one data point per column is tracked per day, but all days are not necessarily tracked,
            # the data will be drawn from a multivariate hypergeometric distribution.
            # A multivariate hypergeometric distribution gives the sampling distribution for small populations.
            # E.g., if one may not have kept track every day - in this case, e.g. for say 10 days
            # recorded over the span of 30 days, we can view this as 10 days randomly sampled from 30 days
            # (This assumes _random_ sampling from the population, which is unlikely in reality, but we can start here)
            # The variance for normal approx to hypergeometric is given by: p*(1-p)*(N-n)/(N-1) for binary variables,
            # where n is number of trials, N is population size.
            # This is equal to the variance * (N-n)/(N-1) in general.
            if ((max_data_counts - counts[self.value_column_name]) < 0).any():
                raise AssertionError(
                    f"Multiple samples not allowed per day when allow_multiple_samples_per_day set to False."
                )
            else:
                stderrs = stdevs[self.value_column_name] * np.sqrt(
                    (max_data_counts - counts[self.value_column_name])
                    / (max_data_counts - 1)
                )

        return df, means, stdevs, stderrs

    def plot_single_data_vs_time(
        self,
        dataframe: pd.DataFrame,
        save_file_name: str,
        ylim_pad: float = 0.5,
        stdevs: bool = True,
        stderr: bool = False,
        maxmins: bool = True,
        tag: str = "",
    ):
        """
        Plot a single data from dataframe vs time.

        :param dataframe: The pandas dataframe corresponding to a single feature, to plot against time.
        :param save_file_name: The file save name.
        :param ylim_pad: extra amount to give +/- the daily max and min when setting ylim.
        :param stdevs: if True, plot standard deviations.
        :param maxmins: if True, plot maxes and mins.
        :param tag: optional tag for end of plot name.
        """
        fig, ax = plt.subplots()

        if maxmins:
            ax.errorbar(
                dataframe.index,
                dataframe.means,
                yerr=[
                    dataframe.means - dataframe.resampled_mins,
                    dataframe.resampled_maxes - dataframe.means,
                ],
                fmt="-o",
                ecolor="red",
                elinewidth=2,
                label="value with max/min (per day) bar",
            )
        if stdevs:
            ax.errorbar(
                dataframe.index,
                dataframe.means,
                yerr=dataframe.stdevs,
                fmt="-o",
                ecolor="blue",
                elinewidth=1.2,
                capsize=2,
                color="blue",
                label="value with std dev bar",
            )
        if stderr:
            ax.errorbar(
                dataframe.index,
                dataframe.means,
                yerr=dataframe.stderrs,
                fmt="-o",
                ecolor="purple",
                elinewidth=1.5,
                capsize=3,
                color="purple",
                label="value with std err bar",
            )
        if not (maxmins or stdevs or stderr):
            ax.plot(dataframe.index, dataframe.means, "-o", color="blue")

        ax.set(
            xlabel="Time",
            ylabel=dataframe.name,
            title=dataframe.name + " by " + self.resample_by.name,
        )
        plt.xticks(rotation=60)
        plt.ylim(
            [
                min(dataframe.resampled_mins) - ylim_pad,
                max(dataframe.resampled_maxes) + ylim_pad,
            ]
        )
        ax.grid()
        ax.legend()
        plt.tight_layout()
        fig.savefig(
            save_file_name
            + "_"
            + self.get_descriptive_string(dataframe.name)
            + tag
            + ".png"
        )
