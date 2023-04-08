from src.utils import *
from src.trainer import Trainer
from src.data import (
    AbstractProcessor,
    AbstractFeatureSelector,
    AbstractTransformer,
)
import inspect
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler as skStandardScaler
from sklearn.preprocessing import OrdinalEncoder
from typing import Type
import shap


class LackDataMaterialRemover(AbstractProcessor):
    """
    Remove materials with fewer data (last 80%).
    """

    def __init__(self):
        super(LackDataMaterialRemover, self).__init__()

    def _fit_transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        m_codes = data.loc[:, "Material_Code"].copy()
        m_cnts_index = list(m_codes.value_counts(ascending=False).index)
        self.lack_data_mat = m_cnts_index[len(m_cnts_index) // 10 * 8 :]
        for m_code in self.lack_data_mat:
            m_codes = data.loc[:, "Material_Code"].copy()
            where_material = m_codes.index[np.where(m_codes == m_code)[0]]
            data = data.drop(where_material)
        return data

    def _transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        for m_code in self.lack_data_mat:
            m_codes = data.loc[:, "Material_Code"].copy()
            where_material = m_codes.index[np.where(m_codes == m_code)[0]]
            data = data.drop(where_material)
        return data


class MaterialSelector(AbstractProcessor):
    """
    Select data with the specified material code. Required arguments:

    m_code: str
        The selected material code.
    """

    def __init__(self):
        super(MaterialSelector, self).__init__()

    def _fit_transform(
        self, data: pd.DataFrame, trainer: Trainer, m_code=None, **kwargs
    ):
        if m_code is None:
            raise Exception('MaterialSelector requires the argument "m_code".')
        m_codes = trainer.df.loc[np.array(data.index), "Material_Code"].copy()
        if m_code not in list(m_codes):
            raise Exception(f"m_code {m_code} not available.")
        where_material = m_codes.index[np.where(m_codes == m_code)[0]]
        data = data.loc[where_material, :]
        self.m_code = m_code
        return data

    def _transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        m_codes = data.loc[:, "Material_Code"].copy()
        if self.m_code not in list(m_codes):
            raise Exception(f"m_code {self.m_code} not available.")
        where_material = m_codes.index[np.where(m_codes == self.m_code)[0]]
        data = data.loc[where_material, :]
        return data


class FeatureValueSelector(AbstractProcessor):
    """
    Select data with the specified feature value. Required arguments:

    feature: str
        The feature that will be filtered.
    value: float
        The specified feature value.
    """

    def __init__(self):
        super(FeatureValueSelector, self).__init__()

    def _fit_transform(
        self,
        data: pd.DataFrame,
        trainer: Trainer,
        feature=None,
        value=None,
        **kwargs,
    ):
        if feature is None or value is None:
            raise Exception(
                'FeatureValueSelector requires arguments "feature" and "value".'
            )
        if value not in list(data[feature]):
            raise Exception(
                f"Value {value} not available for feature {feature}. Select from {data[feature].unique()}"
            )
        where_value = data.index[np.where(data[feature] == value)[0]]
        data = data.loc[where_value, :]
        self.feature, self.value = feature, value
        return data

    def _transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        if self.value not in list(data[self.feature]):
            raise Exception(
                f"Value {self.value} not available for feature {self.feature}. Select from {data[self.feature].unique()}"
            )
        where_value = data.index[np.where(data[self.feature] == self.value)[0]]
        data = data.loc[where_value, :]
        return data


class IQRRemover(AbstractProcessor):
    """
    Remove outliers using IQR strategy. Outliers are those
    out of the range [25-percentile - 1.5 * IQR, 75-percentile + 1.5 * IQR], where IQR = 75-percentile - 25-percentile.
    """

    def __init__(self):
        super(IQRRemover, self).__init__()

    def _fit_transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        print(f"Removing outliers by IQR. Original size: {len(data)}, ", end="")
        for feature in list(trainer.args["feature_names_type"].keys()):
            if pd.isna(data[feature]).all():
                raise Exception(f"All values of {feature} are NaN.")
            Q1 = np.percentile(data[feature].dropna(axis=0), 25, method="midpoint")
            Q3 = np.percentile(data[feature].dropna(axis=0), 75, method="midpoint")
            IQR = Q3 - Q1
            if IQR == 0:
                continue
            upper = data.index[np.where(data[feature] >= (Q3 + 1.5 * IQR))[0]]
            lower = data.index[np.where(data[feature] <= (Q1 - 1.5 * IQR))[0]]

            data = data.drop(upper)
            data = data.drop(lower)
        print(f"Final size: {len(data)}.")
        return data

    def _transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        return data


class StdRemover(AbstractProcessor):
    """
    Remove outliers using standard error strategy. Outliers are those out of the range of 3sigma.
    """

    def __init__(self):
        super(StdRemover, self).__init__()

    def _fit_transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        print(f"Removing outliers by std. Original size: {len(data)}, ", end="")
        for feature in list(trainer.args["feature_names_type"].keys()):
            if pd.isna(data[feature]).all():
                raise Exception(f"All values of {feature} are NaN.")
            m = np.mean(data[feature].dropna(axis=0))
            std = np.std(data[feature].dropna(axis=0))
            if std == 0:
                continue
            upper = data.index[np.where(data[feature] >= (m + 3 * std))[0]]
            lower = data.index[np.where(data[feature] <= (m - 3 * std))[0]]

            data = data.drop(upper)
            data = data.drop(lower)
        print(f"Final size: {len(data)}.")
        return data

    def _transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        return data


class NaNFeatureRemover(AbstractFeatureSelector):
    """
    Remove features that contain no valid value.
    """

    def __init__(self):
        super(NaNFeatureRemover, self).__init__()

    def _get_feature_names_out(self, data, trainer, **kwargs):
        retain_features = []
        all_missing_idx = np.where(
            pd.isna(data[trainer.all_feature_names]).values.all(axis=0)
        )[0]
        for idx, feature in enumerate(trainer.all_feature_names):
            if idx not in all_missing_idx:
                retain_features.append(feature)
        return retain_features


class RFEFeatureSelector(AbstractFeatureSelector):
    """
    Select features using recursive feature elimination, adapted from the implementation of RFECV in sklearn.
    Available arguments:

    n_estimators: int
        The number of trees used in random forests.
    step: int
        The number of eliminated features at each step.
    min_features_to_select: int
        The minimum number of features.
    method: str
        The method of calculating importance. "auto" for default impurity-based method implemented in
        RandomForestRegressor, and "shap" for SHAP value (which may slow down the program but is more accurate).
    """

    def __init__(self):
        super(RFEFeatureSelector, self).__init__()

    def _get_feature_names_out(
        self,
        data,
        trainer,
        n_estimators=100,
        step=1,
        verbose=0,
        min_features_to_select=1,
        method="auto",
        **kwargs,
    ):
        from src.utils.processors.rfecv import ExtendRFECV
        import shap

        cv = KFold(5)

        def importance_getter(estimator, data):
            np.random.seed(0)
            selected_data = data.loc[
                np.random.choice(
                    np.arange(data.shape[0]),
                    size=min(100, data.shape[0]),
                    replace=False,
                ),
                :,
            ]
            return np.mean(
                np.abs(shap.Explainer(estimator)(selected_data).values),
                axis=0,
            )

        rfecv = ExtendRFECV(
            estimator=trainer.get_base_predictor(
                categorical=False,
                n_estimators=100,
                n_jobs=-1,
                random_state=0,
            ),
            step=step,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            min_features_to_select=min_features_to_select,
            n_jobs=-1,
            verbose=verbose,
            importance_getter=importance_getter if method == "shap" else method,
        )
        rfecv.fit(
            data[trainer.all_feature_names],
            data[trainer.label_name].values.flatten(),
        )
        retain_features = list(rfecv.get_feature_names_out())
        return retain_features


class VarianceFeatureSelector(AbstractFeatureSelector):
    """
    Remove features that almost (by a certain fraction) contain an identical value. Required arguments:

    thres: float
        If more than thres * 100 percent of values are the same, the feature is removed.
    """

    def __init__(self):
        super(VarianceFeatureSelector, self).__init__()

    def _get_feature_names_out(self, data, trainer, thres=0.8, **kwargs):
        sel = VarianceThreshold(threshold=(thres * (1 - thres)))
        sel.fit(
            data[trainer.all_feature_names],
            data[trainer.label_name].values.flatten(),
        )
        retain_features = list(sel.get_feature_names_out())
        return retain_features


class CorrFeatureSelector(AbstractFeatureSelector):
    """
    Select features that are not correlated (in the sense of Pearson correlation). Correlated features will be ranked
    by SHAP using RandomForestRegressor, and the feature with the highest importance will be selected.
    Required arguments:

    thres:
        The threshold of pearson correlation.
    n_estimators:
        The number of trees used in random forests.
    """

    def __init__(self):
        super(CorrFeatureSelector, self).__init__()

    def _get_feature_names_out(
        self, data, trainer, thres=0.8, n_estimators=100, **kwargs
    ):
        abs_corr = trainer.cal_corr(imputed=False, features_only=True).abs()
        where_corr = np.where(abs_corr > thres)
        where_corr = [[trainer.cont_feature_names[x] for x in y] for y in where_corr]
        corr_chain = {}

        def add_edge(x, y):
            if x not in corr_chain.keys():
                corr_chain[x] = [y]
            elif y not in corr_chain[x]:
                corr_chain[x].append(y)

        for x, y in zip(*where_corr):
            if x != y:
                add_edge(x, y)
                add_edge(y, x)
        corr_feature = list(corr_chain.keys())
        for x in np.setdiff1d(trainer.cont_feature_names, corr_feature):
            corr_chain[x] = []

        def dfs(visited, graph, node, ls):
            if node not in visited:
                ls.append(node)
                visited.add(node)
                for neighbour in graph[node]:
                    ls = dfs(visited, graph, neighbour, ls)
            return ls

        corr_sets = []
        for x in corr_feature[::-1]:
            if len(corr_sets) != 0:
                for sets in corr_sets:
                    if x in sets:
                        break
                else:
                    corr_sets.append(dfs(set(), corr_chain, x, []))
            else:
                corr_sets.append(dfs(set(), corr_chain, x, []))

        corr_sets = [[x for x in y] for y in corr_sets]

        rf = trainer.get_base_predictor(
            categorical=False, n_estimators=n_estimators, n_jobs=-1, random_state=0
        )
        rf.fit(
            data[trainer.all_feature_names],
            data[trainer.label_name].values.flatten(),
        )

        explainer = shap.Explainer(rf)
        shap_values = explainer(
            data.loc[
                np.random.choice(
                    np.array(data.index), size=min([100, len(data)]), replace=False
                ),
                trainer.all_feature_names,
            ]
        )

        retain_features = list(np.setdiff1d(trainer.cont_feature_names, corr_feature))
        attr = np.mean(np.abs(shap_values.values), axis=0)
        print("Correlated features (Ranked by SHAP):")
        for corr_set in corr_sets:
            set_shap = [attr[trainer.all_feature_names.index(x)] for x in corr_set]
            max_shap_feature = corr_set[set_shap.index(np.max(set_shap))]
            retain_features += [max_shap_feature]
            order = np.array(set_shap).argsort()
            corr_set_dict = {}
            for idx in order[::-1]:
                corr_set_dict[corr_set[idx]] = set_shap[idx]
            print(pretty(corr_set_dict))
        retain_features += trainer.cat_feature_names
        return retain_features


class UnscaledDataRecorder(AbstractTransformer):
    """
    Record unscaled data in the trainer. This processor MUST be inserted before ANY AbstractTransformer (like a
    StandardScaler). The recorded data will be used to generate Trainer.df.
    """

    def __init__(self):
        super(UnscaledDataRecorder, self).__init__()

    def _fit_transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        (
            feature_data,
            categorical_data,
            label_data,
        ) = trainer.divide_from_tabular_dataset(data)

        trainer._unscaled_feature_data = feature_data
        trainer._categorical_data = categorical_data
        trainer._unscaled_label_data = label_data
        return data

    def _transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        (
            feature_data,
            categorical_data,
            label_data,
        ) = trainer.divide_from_tabular_dataset(data)
        trainer._unscaled_feature_data = feature_data
        trainer._categorical_data = categorical_data
        trainer._unscaled_label_data = label_data
        return data

    def zero_slip(self, feature_name, x):
        return 0


class StandardScaler(AbstractTransformer):
    """
    The standard scaler implemented using StandardScaler from sklearn.
    """

    def __init__(self):
        super(StandardScaler, self).__init__()

    def _fit_transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        scaler = skStandardScaler()
        data.loc[:, trainer.cont_feature_names] = scaler.fit_transform(
            data.loc[:, trainer.cont_feature_names]
        ).astype(np.float32)

        self.transformer = scaler
        return data

    def _transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        data.loc[:, trainer.cont_feature_names] = self.transformer.transform(
            data.loc[:, trainer.cont_feature_names]
        ).astype(np.float32)
        return data


class CategoricalOrdinalEncoder(AbstractTransformer):
    """
    The categorical feature encoder that transform string values to unique integer values, implemented using
    OrdinalEncoder from sklearn.
    """

    def __init__(self):
        super(CategoricalOrdinalEncoder, self).__init__()

    def _fit_transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        oe = OrdinalEncoder()
        data.loc[:, trainer.cat_feature_names] = oe.fit_transform(
            data.loc[:, trainer.cat_feature_names]
        ).astype(int)
        for feature, categories in zip(trainer.cat_feature_names, oe.categories_):
            trainer.cat_feature_mapping[feature] = categories
        self.transformer = oe
        return data

    def _transform(self, data: pd.DataFrame, trainer: Trainer, **kwargs):
        try:
            data.loc[:, trainer.cat_feature_names] = self.transformer.transform(
                data.loc[:, trainer.cat_feature_names]
            ).astype(int)
        except:
            try:
                # Categorical features are already transformed.
                self.transformer.inverse_transform(
                    data.loc[:, trainer.cat_feature_names]
                )
                return data
            except:
                raise Exception(
                    f"Categorical features are not compatible with the fitted OrdinalEncoder."
                )
        return data

    def zero_slip(self, feature_name, x):
        return 0


processor_mapping = {}
clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
for name, cls in clsmembers:
    if issubclass(cls, AbstractProcessor):
        processor_mapping[name] = cls


def get_data_processor(name: str) -> Type[AbstractProcessor]:
    if name not in processor_mapping.keys():
        raise Exception(f"Data processor {name} not implemented.")
    elif not issubclass(processor_mapping[name], AbstractProcessor):
        raise Exception(f"{name} is not the subclass of AbstractProcessor.")
    else:
        return processor_mapping[name]
