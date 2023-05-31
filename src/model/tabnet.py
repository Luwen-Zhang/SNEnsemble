from src.utils import *
from src.model import AbstractModel
from skopt.space import Real, Integer


class TabNet(AbstractModel):
    def _get_program_name(self):
        return "TabNet"

    def _get_model_names(self):
        return ["TabNet"]

    def _new_model(self, model_name, verbose, **kwargs):
        from pytorch_tabnet.tab_model import TabNetRegressor

        def extract_params(**kwargs):
            params = {}
            optim_params = {}
            batch_size = 32
            for key, value in kwargs.items():
                if key in [
                    "n_d",
                    "n_a",
                    "n_steps",
                    "gamma",
                    "n_independent",
                    "n_shared",
                ]:
                    params[key] = value
                elif key == "batch_size":
                    batch_size = int(value)
                else:
                    optim_params[key] = value
            return params, optim_params, batch_size

        params, optim_params, batch_size = extract_params(**kwargs)

        model = TabNetRegressor(
            verbose=20 if verbose else 0, optimizer_params=optim_params
        )

        model.set_params(**params)
        return model

    def _train_data_preprocess(self):
        data = self.trainer.datamodule
        cont_feature_names = self.trainer.cont_feature_names
        X_train = data.X_train[cont_feature_names].values.astype(np.float32)
        X_val = data.X_val[cont_feature_names].values.astype(np.float32)
        X_test = data.X_test[cont_feature_names].values.astype(np.float32)
        y_train = data.y_train.astype(np.float32)
        y_val = data.y_val.astype(np.float32)
        y_test = data.y_test.astype(np.float32)

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
        }

    def _data_preprocess(self, df, derived_data, model_name):
        return df[self.trainer.cont_feature_names].values.astype(np.float32)

    def _train_single_model(
        self,
        model,
        epoch,
        X_train,
        y_train,
        X_val,
        y_val,
        verbose,
        warm_start,
        in_bayes_opt,
        **kwargs,
    ):
        eval_set = [(X_val, y_val)]

        model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            max_epochs=epoch,
            patience=self.trainer.args["patience"],
            loss_fn=self.trainer.get_loss_fn(),
            eval_metric=[self.trainer.args["loss"]],
            batch_size=int(kwargs["batch_size"]),
            warm_start=warm_start,
        )

    def _pred_single_model(self, model, X_test, verbose, **kwargs):
        return model.predict(X_test).reshape(-1, 1)

    def _space(self, model_name):
        return [
            Integer(low=4, high=64, prior="uniform", name="n_d", dtype=int),  # 8
            Integer(low=4, high=64, prior="uniform", name="n_a", dtype=int),  # 8
            Integer(low=3, high=10, prior="uniform", name="n_steps", dtype=int),  # 3
            Real(low=1.0, high=2.0, prior="uniform", name="gamma"),  # 1.3
            Integer(
                low=1, high=5, prior="uniform", name="n_independent", dtype=int
            ),  # 2
            Integer(low=1, high=5, prior="uniform", name="n_shared", dtype=int),  # 2
        ] + self.trainer.SPACE

    def _initial_values(self, model_name):
        return {
            "n_d": 8,
            "n_a": 8,
            "n_steps": 3,
            "gamma": 1.3,
            "n_independent": 2,
            "n_shared": 2,
            "lr": self.trainer.args["lr"],
            "weight_decay": self.trainer.args["weight_decay"],
            "batch_size": self.trainer.args["batch_size"],
        }
