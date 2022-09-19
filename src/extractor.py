from random import Random
import pandas as pd
import numpy as np


class Extractor():
    def __init__(self, data_path):
        '''
        The automatic article data extractor class.
        :param data_path: The path of the formatted .xlsx file.
        '''
        self.data_path = data_path

    def read(self):
        '''
        Read the formatted .xlsx file from data_path.
        :return: None
        '''
        self.df = pd.read_excel(self.data_path, engine='openpyxl', sheet_name=None)

        # Read sheets and classify
        self.sheet_names = list(self.df.keys())

        self.properties = []
        self.scatters = []
        self.curves = []
        for sheet_name in self.sheet_names:
            if 'Property' in sheet_name:
                self.properties.append(sheet_name)
            elif 'Scatter' in sheet_name:
                self.scatters.append(sheet_name)
            elif 'Curve' in sheet_name:
                self.curves.append(sheet_name)
            elif sheet_name != 'Assignment':
                print('Sheet not identified:', sheet_name)

        # Identify feature names in scatters and properties.
        self.scatter_feature_names = []
        for scatter in self.scatters:
            self.scatter_feature_names += list(self.df[scatter].columns)

        self.feature_names = self.scatter_feature_names.copy()
        for property in self.properties:
            self.feature_names += list(self.df[property].columns)[1:]
        self.feature_names = list(sorted(set(self.feature_names), key=self.feature_names.index))

        assign = self.df['Assignment']

        data = pd.DataFrame(columns=list(assign.columns) + self.feature_names)

        # For each scatter to be assigned, create a dataframe tmp_df that contains all feature values
        for row in assign.iterrows():
            # Find all properties that assigned to the scatter
            prop_to_assign = [x for x in list(assign.columns)[1:] if pd.notna(row[1][x])]

            # Find all features related to the scatter
            assigned_features = []
            scatter_name = row[1]['Scatters']
            for property in prop_to_assign:
                assigned_property_feature = list(self.df[property].columns)[1:]
                assigned_features += assigned_property_feature

            # Create the dataframe for the scatter
            tmp_df = pd.DataFrame(columns=list(row[1].keys()) + list(self.df[scatter_name].columns) + assigned_features)

            # Assign scatter values
            scatter_df = self.df[scatter_name]
            for col_name in scatter_df.columns:
                tmp_df[col_name] = scatter_df[col_name]

            # For each property, extract values and assign them to all scatter points
            for property in prop_to_assign:
                assigned_property_feature = list(self.df[property].columns)[1:]
                assigned_property_index = list(self.df[property]['Property']).index(row[1][property])
                assigned_property_value = self.df[property].loc[assigned_property_index, assigned_property_feature]

                for row_idx in tmp_df.index:
                    tmp_df.loc[row_idx, assigned_property_feature] = assigned_property_value

            # Finally, assign information to the dataframe
            for row_idx in tmp_df.index:
                tmp_df.loc[row_idx, row[1].keys()] = row[1].values

            data = pd.concat([data, tmp_df], axis=0, ignore_index=True)

            data.reset_index(drop=True)

        # Get curve relations
        self.curve_relation = []
        for curve in self.curves:
            curve_data = self.df[curve]
            self.curve_relation.append(list(curve_data.columns))

        # At the end of this part, for each data point(row in the dataframe), feature values are probably missing due to unrelated features in different scatters
        self.data = data
        self.initial_data = data.copy()

    def apply_transform(self, column_from, column_to, transform_func, **kargs):
        '''
        Transform one feature to another according to a function. Only NaN values in column_to will be filled.
        :param column_from: The input feature of transform_func
        :param column_to: The output feature of transform_func
        :param transform_func: User defined transformation function handler.
        :param kargs: Parameters for transform_func
        :return: A boolean value representing whether any value has been transformed.
        '''
        # transform_idx represents indexes of feature values to be transformed in column_to
        # new_feature is a boolean represents whether a new feature would be created.
        transform_idx, new_feature = self.check_available_transform(column_from, column_to)

        if len(transform_idx) == 0:
            return False
        else:
            # Apply func one by one
            for idx in transform_idx:
                self.data.loc[idx, column_to] = transform_func(self.data.loc[idx, column_from], **kargs)
            if new_feature:
                self.feature_names.append(column_to)
            return True

    def check_available_transform(self, column_from, column_to):
        '''
        Check non-NaN values in features encountered in apply_transform()
        :param column_from: The input feature of transform_func of apply_transform()
        :param column_to: The output feature of transform_func of apply_transform()
        :return: A list (ndarray) of transformable indexes; A boolean value representing whether a new feature will be created.
        '''
        # column_from must exist in feature_names
        if column_from not in self.feature_names:
            return [], False

        # Indexes of values (not NaN)
        column_from_exist = np.where(pd.notna(self.data[column_from]))

        if column_to not in self.feature_names:
            # If column_to does not exist, column_from_exist contains all indexes to be transformed.
            return column_from_exist[0], True
        else:
            # Only when self.data.loc[idx,column_to] is NaN will the transformation be applied.
            column_to_exist = np.where(pd.notna(self.data[column_to]))
            transform_idx = np.setdiff1d(column_from_exist, column_to_exist)
            return transform_idx, False

    def fill_na(self, column, criterion, remove_na_axis):
        '''
        Fill NaN values of one feature using a machine learning model, taking other features as predictors.
        :param column: The feature to be filled.
        :param criterion: Machine learning model for NaN value prediction.
        :param remove_na_axis: Indicates deleting rows(0) or columns(1) when other features contain NaN values.
        :return: False if nothing has been filled, otherwise None.
        '''
        if column not in self.feature_names:
            print('Feature does not exist.')
            return False

        print('Fill NaN of feature', column)

        train_features = self.feature_names.copy()
        train_features.remove(column)

        data = self.data.copy()[train_features]

        # Some features may contain NaN values, which should be dropped
        data = data.dropna(axis=remove_na_axis)
        if remove_na_axis == 1:
            # Reset train_features since some columns might be dropped
            train_features = list(data.columns)

        column_data = self.data.copy().loc[data.index, column]
        train_index = np.where(pd.notna(column_data))[0]
        pred_index = np.setdiff1d(data.index, train_index)

        if len(pred_index) == 0:
            print('\tNo NaN to be filled.')
            return False

        train_data = np.array(data.loc[train_index, train_features], dtype=np.float32)
        train_label = np.array(column_data[train_index], dtype=np.float32)
        pred_data = np.array(data.loc[pred_index, train_features], dtype=np.float32)

        if criterion == 'RandomForest':
            from sklearn.ensemble import RandomForestRegressor

            rf = RandomForestRegressor(n_jobs=-1)

            rf.fit(train_data, train_label)
            print('\tPredictors: ', train_features)
            print(
                f'\t{train_data.shape[0]} data as training set, {pred_data.shape[0]} to be predict, R2 score {rf.score(train_data, train_label)}.')

            pred_label = rf.predict(pred_data)

            # Fill NaN here
            self.data.loc[pred_index, column] = pred_label

        else:
            print('\tCriterion not implemented')
            return False

    def interpolate_from_curves(self):
        '''
        Automatically interpolate missing values using Curve(s).
        :return: None
        '''

        def check_available_interp():
            # Return a ndarray that represents available transform numbers of curve relations and directions (x->y or y->x)
            available_interp = np.zeros((len(self.curve_relation), 2))
            for idx, relation in enumerate(self.curve_relation):
                available_interp[idx, 0] = len(self.check_available_transform(relation[0], relation[1])[0])
                available_interp[idx, 1] = len(self.check_available_transform(relation[1], relation[0])[0])
            return available_interp

        available_interp = check_available_interp()
        while np.sum(available_interp) != 0:
            # Until all available interpolations are eliminated.
            best_interp = np.where(available_interp == np.max(available_interp))
            best_interp_relation = best_interp[0][0]
            best_interp_from = best_interp[1][0]

            self.interpolate_from_curve(self.curves[best_interp_relation], best_interp_from)

            available_interp = check_available_interp()

    def interpolate_from_curve(self, curve_name, direction):
        '''
        Interpolate using specific Curve data.
        :param curve_name: The name of Curve sheet.
        :param direction: 0 to interpolate y from x. 1 to interpolate x from y.
        :return: None
        '''
        curve_data = self.df[curve_name]
        column_from = curve_data.columns[direction]
        column_to = curve_data.columns[1 - direction]
        curve_x = curve_data[column_from].values
        curve_y = curve_data[column_to].values

        def interpolation(x):
            # This is a example of linear interpolation.
            if x < np.min(curve_x) or x > np.max(curve_x):
                print('Interpolated data exceeds the curve range.')
            return np.interp(x, curve_x, curve_y)

        status = self.apply_transform(column_from, column_to, interpolation)
        if status:
            print(f'Interpolate feature \"{column_to}\" from feature \"{column_from}\" using \"{curve_name}\"')
        else:
            print(f'Not interpolated to feature \"{column_to}\" from feature \"{column_from}\"')

    def to_csv(self, path, **kargs):
        self.data.to_csv(path, **kargs)

    def to_excel(self, path, **kargs):
        self.data.to_excel(path, engine='openpyxl', **kargs)

    def restore_data(self):
        self.data = self.initial_data.copy()
