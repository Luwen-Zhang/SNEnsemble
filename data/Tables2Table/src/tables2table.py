from random import Random
import pandas as pd
import numpy as np


class Tables2Table:
    def __init__(self, data_path):
        """
        The automatic article data extractor class.
        :param data_path: The path of the formatted .xlsx file.
        """
        self.data_path = data_path

    def read(self):
        """
        Read the formatted .xlsx file from data_path.
        :return: None
        """
        self._df = pd.read_excel(self.data_path, engine="openpyxl", sheet_name=None)

        # Read sheets and classify
        self._sheet_names = list(self._df.keys())

        self._properties = []
        self._scatters = []
        self._curves = []
        self._tables = []
        self._curves_prop = {}
        self._curves_df = {}
        for sheet_name in self._sheet_names:
            if "Property" in sheet_name:
                self._properties.append(sheet_name)
            elif "Scatter" in sheet_name:
                self._scatters.append(sheet_name)
            elif "Curve" in sheet_name:
                self._curves.append(sheet_name)
            elif "Table" in sheet_name:
                self._tables.append(sheet_name)
            elif sheet_name != "Assignment":
                print("Sheet not identified:", sheet_name)

        # Identify feature names in scatters and properties.
        self._scatter_feature_names = []
        for scatter in self._scatters:
            self._scatter_feature_names += list(self._df[scatter].columns)

        self._table_feature_names = []
        for table in self._tables:
            self._table_feature_names += [
                x for x in list(self._df[table].columns) if x not in self._properties
            ]

        self._feature_names = (
            self._scatter_feature_names.copy() + self._table_feature_names.copy()
        )
        for property in self._properties:
            self._feature_names += list(self._df[property].columns)[1:]
        self._feature_names = list(
            sorted(set(self._feature_names), key=self._feature_names.index)
        )

        if "Assignment" in list(self._df.keys()):
            assign = self._df["Assignment"]

            data = pd.DataFrame(
                columns=(
                    list(assign.columns) + ["Table"]
                    if len(self._tables) != 0
                    else [] + self._feature_names
                )
            )
        else:
            assign = pd.DataFrame(columns=[""])

            data = pd.DataFrame(
                columns=(
                    ["Table"] if len(self._tables) != 0 else [] + self._feature_names
                )
            )

        # For each scatter to be assigned, create a dataframe tmp_df that contains all feature values
        for row in assign.iterrows():
            # Find all properties that assigned to the scatter
            if len(assign.columns) > 1:
                prop_to_assign = [
                    x for x in list(assign.columns)[1:] if pd.notna(row[1][x])
                ]
            else:
                prop_to_assign = []

            # Find all features related to the scatter
            assigned_features = []
            scatter_name = row[1]["Scatter"]
            for property in prop_to_assign:
                assigned_property_feature = list(self._df[property].columns)[1:]
                assigned_features += assigned_property_feature

            # Create the dataframe for the scatter
            tmp_df = pd.DataFrame(
                columns=list(row[1].keys())
                + list(self._df[scatter_name].columns)
                + assigned_features
            )

            # Assign scatter values
            scatter_df = self._df[scatter_name]
            for col_name in scatter_df.columns:
                tmp_df[col_name] = scatter_df[col_name]

            # For each property, extract values and assign them to all scatter points
            for property in prop_to_assign:
                assigned_property_feature = list(self._df[property].columns)[1:]
                assigned_property_index = list(self._df[property]["Property"]).index(
                    row[1][property]
                )
                assigned_property_value = self._df[property].loc[
                    assigned_property_index, assigned_property_feature
                ]

                for row_idx in tmp_df.index:
                    tmp_df.loc[row_idx, assigned_property_feature] = (
                        assigned_property_value
                    )

            # Finally, assign information to the dataframe
            for row_idx in tmp_df.index:
                tmp_df.loc[row_idx, row[1].keys()] = row[1].values

            if "Scatter" in scatter_name:
                data = pd.concat([data, tmp_df], axis=0, ignore_index=True)

                data.reset_index(drop=True, inplace=True)
            else:
                tmp = row[1].copy()
                tmp.pop("Scatter")
                del tmp_df["Scatter"]
                self._curves_prop[scatter_name] = tmp
                self._curves_df[scatter_name] = tmp_df

        # Identify table data
        for table in self._tables:
            df_table = self._df[table]
            table_props = list(
                set([x for x in list(df_table.columns) if x in self._properties])
            )
            table_props_features = []
            for property in table_props:
                table_props_features += list(self._df[property].columns)[1:]
            table_props_features = list(set(table_props_features))

            table_feature_no_props = np.setdiff1d(
                np.array(df_table.columns), table_props
            )
            df_table_props = df_table[table_props].copy()
            df_table.loc[:, table_props_features] = np.nan
            for row_idx, row in enumerate(df_table.iterrows()):
                row_props_name = df_table_props.loc[row_idx, :].values
                row_props = list(df_table_props.columns)
                for property, property_name in zip(row_props, row_props_name):
                    assigned_property_feature = list(self._df[property].columns)[1:]
                    try:
                        assigned_property_index = list(
                            self._df[property]["Property"]
                        ).index(property_name)
                    except:
                        # No such property
                        continue
                    assigned_property_value = self._df[property].loc[
                        assigned_property_index, assigned_property_feature
                    ]
                    isna_feature = np.where(
                        [
                            np.isnan(x)
                            for x in df_table.loc[
                                row_idx, assigned_property_feature
                            ].values
                        ]
                    )[0]
                    df_table.loc[
                        row_idx, np.array(assigned_property_feature)[isna_feature]
                    ] = assigned_property_value

            df_table["Table"] = table
            data = pd.concat([data, df_table], axis=0, ignore_index=True)
            data.reset_index(drop=True, inplace=True)

        # Get curve relations
        self._curve_relation = []
        for curve in self._curves:
            curve_data = self._df[curve]
            self._curve_relation.append(list(curve_data.columns))

        # At the end of this part, for each data point(row in the dataframe), feature values are probably missing due to unrelated features in different scatters
        self._data = data
        self._initial_data = data.copy()
        self._initial_feature = self._feature_names.copy()

    def get_data(self):
        return self._data.copy()

    def get_curves(self):
        return self._curves.copy()

    def get_curve_relation(self):
        return self._curve_relation.copy()

    def get_df(self):
        return self._df.copy()

    def get_properties(self):
        return self._properties.copy()

    def get_scatters(self):
        return self._scatters.copy()

    def get_feature_names(self):
        return self._feature_names.copy()

    def apply_transform(
        self, column_from, column_to, transform_func, indexes=[], **kargs
    ):
        """
        Transform one feature to another according to a function. Only NaN values in column_to will be filled.
        :param column_from: The input feature of transform_func
        :param column_to: The output feature of transform_func
        :param transform_func: User defined transformation function handler.
        :param indexes: Indexes of values to be transformed.
        :param kargs: Parameters for transform_func
        :return: Indexes of transformed values.
        """
        # transform_idx represents indexes of feature values to be transformed in column_to
        # new_feature is a boolean represents whether a new feature would be created.
        transform_idx, new_feature = self.check_available_transform(
            column_from, column_to
        )

        if len(indexes) != 0:
            transform_idx = np.intersect1d(transform_idx, indexes)

        if len(transform_idx) != 0:
            # Apply func one by one
            for idx in transform_idx:
                self._data.loc[idx, column_to] = transform_func(
                    self._data.loc[idx, column_from], **kargs
                )
            if new_feature:
                self._feature_names.append(column_to)

        return transform_idx

    def check_available_transform(self, column_from, column_to):
        """
        Check non-NaN values in features encountered in apply_transform()
        :param column_from: The input feature of transform_func of apply_transform()
        :param column_to: The output feature of transform_func of apply_transform()
        :return: A list (ndarray) of transformable indexes; A boolean value representing whether a new feature will be created.
        """
        # column_from must exist in feature_names
        if column_from not in self._feature_names:
            return [], False

        # Indexes of values (not NaN)
        column_from_exist = np.where(pd.notna(self._data[column_from]))

        if column_to not in self._feature_names:
            # If column_to does not exist, column_from_exist contains all indexes to be transformed.
            return column_from_exist[0], True
        else:
            # Only when self._data.loc[idx,column_to] is NaN will the transformation be applied.
            column_to_exist = np.where(pd.notna(self._data[column_to]))
            transform_idx = np.setdiff1d(column_from_exist, column_to_exist)
            return transform_idx, False

    def fill_na(self, column, criterion, remove_na_axis):
        """
        Fill NaN values of one feature using a machine learning model, taking other features as predictors.
        :param column: The feature to be filled.
        :param criterion: Machine learning model for NaN value prediction.
        :param remove_na_axis: Indicates deleting rows(0) or columns(1) when other features contain NaN values.
        :return: False if nothing has been filled, otherwise None.
        """
        if column not in self._feature_names:
            print("Feature does not exist.")
            return False

        print("Fill NaN of feature", column)

        train_features = self._feature_names.copy()
        train_features.remove(column)

        data = self._data.copy()[train_features]

        # Some features may contain NaN values, which should be dropped
        data = data.dropna(axis=remove_na_axis)
        if remove_na_axis == 1:
            # Reset train_features since some columns might be dropped
            train_features = list(data.columns)

        column_data = self._data.copy().loc[data.index, column]
        train_index = column_data.index[pd.notna(column_data)]
        pred_index = np.setdiff1d(data.index, train_index)

        if len(pred_index) == 0:
            print("\tNo NaN to be filled.")
            return False

        train_data = np.array(data.loc[train_index, train_features], dtype=np.float32)
        train_label = np.array(column_data[train_index], dtype=np.float32)
        pred_data = np.array(data.loc[pred_index, train_features], dtype=np.float32)

        if criterion == "RandomForest":
            from sklearn.ensemble import RandomForestRegressor

            rf = RandomForestRegressor(n_jobs=-1)

            rf.fit(train_data, train_label)
            print(
                "\tPredictors: ",
                train_features,
                "\n\tTraining set",
                list(train_index),
                "\n\tPred set",
                list(pred_index),
            )
            print(f"\tR2 score {rf.score(train_data, train_label):.5f}.")

            pred_label = rf.predict(pred_data)

            # Fill NaN here
            self._data.loc[pred_index, column] = pred_label

        else:
            print("\tCriterion not implemented")
            return False

    def interpolate_from_curves(self, ignore_property=True):
        """
        Automatically interpolate missing values using Curve(s).
        :param ignore_property: Whether properties of curves should be ignored.
        :return: None
        """

        def check_available_interp():
            # Return a ndarray that represents available transform numbers of curve relations and directions (x->y or y->x)
            available_interp = np.zeros((len(self._curve_relation), 2))
            for idx, relation in enumerate(self._curve_relation):
                if ignore_property:
                    available_interp[idx, 0] = len(
                        self.check_available_transform(relation[0], relation[1])[0]
                    )
                    available_interp[idx, 1] = len(
                        self.check_available_transform(relation[1], relation[0])[0]
                    )
                else:
                    curve_prop_fit = self.check_curve_property_fit(self._curves[idx])
                    available_interp[idx, 0] = len(
                        np.intersect1d(
                            self.check_available_transform(relation[0], relation[1])[0],
                            curve_prop_fit,
                        )
                    )
                    available_interp[idx, 1] = len(
                        np.intersect1d(
                            self.check_available_transform(relation[1], relation[0])[0],
                            curve_prop_fit,
                        )
                    )
            return available_interp

        available_interp = check_available_interp()
        while np.sum(available_interp) != 0:
            # Until all available interpolations are eliminated.
            best_interp = np.where(available_interp == np.max(available_interp))
            best_interp_relation = best_interp[0][0]
            best_interp_from = best_interp[1][0]

            self.interpolate_from_curve(
                self._curves[best_interp_relation],
                best_interp_from,
                ignore_property=ignore_property,
            )

            available_interp = check_available_interp()

    def check_curve_property_fit(self, curve_name):
        # Find available indexes where curve properties fit.
        curve_df = self._curves_df[curve_name].copy()
        curve_relation = self._curve_relation[self._curves.index(curve_name)]
        prop_feature = list(curve_df.columns)
        prop_feature = [
            x
            for x in prop_feature
            if x not in curve_relation and x not in self._properties
        ]
        prop_data = curve_df.loc[0, prop_feature]

        prop_data_all = self._data[prop_data.keys()]

        flag = np.zeros((prop_data_all.shape[0], 1)).astype(bool)
        for row in prop_data_all.iterrows():
            for pair in zip(prop_data, row[1]):
                if np.isnan(pair[0]) or np.isnan(pair[1]):
                    pass
                elif np.abs(pair[0] - pair[1]) > 1e-8:
                    break
            else:
                flag[row[0], 0] = True

        indexes = np.where(flag)[0]
        return indexes

    def interpolate_from_curve(self, curve_name, direction, ignore_property=True):
        """
        Interpolate using specific Curve data.
        :param curve_name: The name of Curve sheet.
        :param direction: 0 to interpolate y from x. 1 to interpolate x from y.
        :param ignore_property: Whether properties of curves should be ignored.
        :return: None
        """
        curve_data = self._df[curve_name].copy()
        column_from = curve_data.columns[direction]
        column_to = curve_data.columns[1 - direction]
        curve_x = curve_data[column_from].values
        curve_y = curve_data[column_to].values

        if ignore_property == False:
            indexes = self.check_curve_property_fit(curve_name)
        else:
            indexes = []

        def interpolation(x):
            # This is a example of linear interpolation.
            if x < np.min(curve_x) or x > np.max(curve_x):
                print("Interpolated data exceeds the curve range.")
            return np.interp(x, curve_x, curve_y)

        transform_idx = self.apply_transform(
            column_from, column_to, interpolation, indexes
        )
        print(
            f'Interpolated feature "{column_to}" from feature "{column_from}" using "{curve_name}"'
        )
        print(f"\tIndexes {list(transform_idx)}")

    def remove_n(self):
        self._data.columns = [x.replace("\n", " ") for x in self._data.columns]

    def to_csv(self, path, **kargs):
        self.remove_n()
        self._data.to_csv(path, **kargs)

    def to_excel(self, path, **kargs):
        self._data.to_excel(path, engine="openpyxl", **kargs)

    def restore_data(self):
        self._data = self._initial_data.copy()
        self._feature_names = self._initial_feature.copy()


def create_example(data_path):
    np.random.seed(0)

    property_1 = pd.DataFrame(
        {
            "Property": ["P1", "P2", "P3"],
            "A": np.random.randn(3),
            "B": np.random.randn(3),
        }
    )
    property_2 = pd.DataFrame(
        {
            "Property": ["P4", "P5", "P6"],
            "B": np.random.randn(3),
            "D": np.random.randn(3),
        }
    )

    property_1.loc[0, "B"] = np.nan

    scatter_1 = pd.DataFrame({"D": np.random.randn(5), "E": np.random.randn(5)})
    scatter_2 = pd.DataFrame({"A": np.random.randn(5), "C": np.random.randn(5)})
    scatter_3 = pd.DataFrame({"D": np.random.randn(5), "E": np.random.randn(5)})

    curve_1 = pd.DataFrame({"C": np.linspace(-2, 2, 10), "D": np.linspace(-2, 2, 10)})
    curve_2 = pd.DataFrame({"A": np.linspace(-2, 2, 10), "C": np.linspace(-2, 2, 10)})
    curve_3 = pd.DataFrame({"A": np.linspace(-2, 2, 10), "F": np.linspace(-2, 2, 10)})

    assign = pd.DataFrame(
        {
            "Scatter": [
                "Scatter 1",
                "Scatter 2",
                "Scatter 3",
                "Curve 1",
                "Curve 2",
                "Curve 3",
            ],
            "Property 1": ["P1", np.nan, "P2", "P1", np.nan, np.nan],
            "Property 2": [np.nan, "P4", np.nan, np.nan, "P4", np.nan],
        }
    )

    table_1 = pd.DataFrame(
        {
            "Property 1": ["P1", "P2", "P2"],
            "Property 2": ["P4", "P4", "P5"],
            "C": np.random.randn(3),
        }
    )

    with pd.ExcelWriter(data_path, engine="openpyxl") as writer:
        property_1.to_excel(writer, "Property 1", index=False)
        property_2.to_excel(writer, "Property 2", index=False)
        scatter_1.to_excel(writer, "Scatter 1", index=False)
        scatter_2.to_excel(writer, "Scatter 2", index=False)
        scatter_3.to_excel(writer, "Scatter 3", index=False)
        curve_1.to_excel(writer, "Curve 1", index=False)
        curve_2.to_excel(writer, "Curve 2", index=False)
        curve_3.to_excel(writer, "Curve 3", index=False)
        assign.to_excel(writer, "Assignment", index=False)
        table_1.to_excel(writer, "Table 1", index=False)
