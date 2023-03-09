from src.utils import *
from src.model import AbstractModel
from src.trainer import save_trainer


class AutoGluon(AbstractModel):
    def __init__(self, trainer=None, program=None, model_subset=None):
        super(AutoGluon, self).__init__(
            trainer, program=program, model_subset=model_subset
        )

    def _get_program_name(self):
        return "AutoGluon"

    def _train(
        self,
        verbose: bool = False,
        model_subset: list = None,
        debug_mode: bool = False,
        dump_trainer=True,
        warm_start=False,
        **kwargs,
    ):
        disable_tqdm()

        if model_subset is not None:
            warnings.warn(
                f"AutoGluon does not support training models separately, but a model_subset is passed to AutoGluon.",
                category=UserWarning,
            )
        warnings.simplefilter(action="ignore", category=UserWarning)
        from autogluon.tabular import TabularPredictor
        from autogluon.features.generators import PipelineFeatureGenerator
        from autogluon.features.generators.category import CategoryFeatureGenerator
        from autogluon.features.generators.identity import IdentityFeatureGenerator
        from autogluon.common.features.feature_metadata import FeatureMetadata
        from autogluon.common.features.types import R_INT, R_FLOAT

        (
            tabular_dataset,
            cont_feature_names,
            cat_feature_names,
            label_name,
        ) = self.trainer.get_tabular_dataset()
        tabular_dataset = self.trainer.categories_inverse_transform(tabular_dataset)
        predictor = TabularPredictor(
            label=label_name[0], path=self.root, problem_type="regression"
        )
        feature_metadata = {}
        for feature in cont_feature_names:
            feature_metadata[feature] = "float"
        for feature in cat_feature_names:
            feature_metadata[feature] = "object"
        feature_generator = PipelineFeatureGenerator(
            generators=[
                [
                    IdentityFeatureGenerator(
                        infer_features_in_args=dict(valid_raw_types=[R_INT, R_FLOAT]),
                        feature_metadata_in=FeatureMetadata(feature_metadata),
                    ),
                    CategoryFeatureGenerator(
                        feature_metadata_in=FeatureMetadata(feature_metadata)
                    ),
                ]
            ]
        )
        with HiddenPrints(disable_logging=True if not verbose else False):
            predictor.fit(
                tabular_dataset.loc[self.trainer.train_indices, :],
                tuning_data=tabular_dataset.loc[self.trainer.val_indices, :],
                presets="best_quality"
                if not debug_mode
                else "medium_quality_faster_train",
                hyperparameter_tune_kwargs="bayesopt"
                if (not debug_mode) and self.trainer.bayes_opt
                else None,
                use_bag_holdout=True,
                verbosity=0 if not verbose else 2,
                feature_generator=feature_generator,
            )
        self.leaderboard = predictor.leaderboard(
            tabular_dataset.loc[self.trainer.test_indices, :], silent=True
        )
        self.leaderboard.to_csv(self.root + "leaderboard.csv")
        self.model = predictor
        enable_tqdm()
        warnings.simplefilter(action="default", category=UserWarning)
        if dump_trainer:
            save_trainer(self.trainer)

    def _predict(self, df: pd.DataFrame, model_name, derived_data=None, **kwargs):
        return self.model.predict(
            df[self.trainer.all_feature_names], model=model_name, **kwargs
        )

    def _get_model_names(self):
        return list(self.leaderboard["model"])
