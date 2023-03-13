from src.utils import *
from src.model import AbstractModel
from typing import Optional, Dict
from pytorch_widedeep.callbacks import Callback
import src


class WideDeep(AbstractModel):
    def _get_program_name(self):
        return "WideDeep"

    def _space(self, model_name):
        return self.trainer.SPACE

    def _initial_values(self, model_name):
        return self.trainer.chosen_params

    def _new_model(self, model_name, verbose, **kwargs):
        from pytorch_widedeep.models import (
            WideDeep,
            TabMlp,
            TabResnet,
            TabTransformer,
            TabNet,
            SAINT,
            ContextAttentionMLP,
            SelfAttentionMLP,
            FTTransformer,
            TabPerceiver,
            TabFastFormer,
        )
        from pytorch_widedeep import Trainer as wd_Trainer
        from pytorch_widedeep.callbacks import EarlyStopping

        cont_feature_names = self.trainer.cont_feature_names
        cat_feature_names = self.trainer.cat_feature_names

        args = dict(
            column_idx=self.tab_preprocessor.column_idx,
            continuous_cols=cont_feature_names,
            cat_embed_input=self.tab_preprocessor.cat_embed_input
            if len(cat_feature_names) != 0
            else None,
        )

        if model_name == "TabTransformer":
            args["embed_continuous"] = True if len(cat_feature_names) == 0 else False

        mapping = {
            "TabMlp": TabMlp,
            "TabResnet": TabResnet,
            "TabTransformer": TabTransformer,
            "TabNet": TabNet,
            "SAINT": SAINT,
            "ContextAttentionMLP": ContextAttentionMLP,
            "SelfAttentionMLP": SelfAttentionMLP,
            "FTTransformer": FTTransformer,
            "TabPerceiver": TabPerceiver,
            "TabFastFormer": TabFastFormer,
        }

        tab_model = mapping[model_name](**args)
        model = WideDeep(deeptabular=tab_model)

        optimizer = torch.optim.Adam(
            model.deeptabular.parameters(),
            lr=kwargs["lr"],
            weight_decay=kwargs["weight_decay"],
        )

        wd_trainer = wd_Trainer(
            model,
            objective="regression",
            verbose=0,
            callbacks=[
                EarlyStopping(
                    patience=self.trainer.static_params["patience"],
                    verbose=1 if verbose else 0,
                    restore_best_weights=True,
                ),
                WideDeepCallback(total_epoch=self.total_epoch, verbose=verbose),
            ],
            optimizers={"deeptabular": optimizer} if self.trainer.bayes_opt else None,
            device=self.trainer.device,
        )
        return wd_trainer

    def _train_data_preprocess(
        self,
        X_train,
        D_train,
        y_train,
        X_val,
        D_val,
        y_val,
        X_test,
        D_test,
        y_test,
    ):
        from pytorch_widedeep.preprocessing import TabPreprocessor
        from pandas._config import option_context

        cont_feature_names = self.trainer.cont_feature_names
        cat_feature_names = self.trainer.cat_feature_names
        tab_preprocessor = TabPreprocessor(
            continuous_cols=cont_feature_names,
            cat_embed_cols=cat_feature_names if len(cat_feature_names) != 0 else None,
        )
        with option_context("mode.chained_assignment", None):
            X_tab_train = tab_preprocessor.fit_transform(X_train)
            X_tab_val = tab_preprocessor.transform(X_val)
            X_tab_test = tab_preprocessor.transform(X_test)
        self.tab_preprocessor = tab_preprocessor
        return (
            X_tab_train,
            D_train,
            y_train,
            X_tab_val,
            D_val,
            y_val,
            X_tab_test,
            D_test,
            y_test,
        )

    def _train_single_model(
        self,
        model,
        epoch,
        X_train,
        D_train,
        y_train,
        X_val,
        D_val,
        y_val,
        verbose,
        warm_start,
        in_bayes_opt,
        **kwargs,
    ):
        """
        pytorch_widedeep uses an approximated loss calculation procedure that calculates the average loss
        across batches, which is not what we do (in a precise way for MSE) at the end of training and makes
        results from the callback differ from our final metrics.
        """
        model.fit(
            X_train={"X_tab": X_train, "target": y_train},
            X_val={"X_tab": X_val, "target": y_val},
            n_epochs=epoch,
            batch_size=int(kwargs["batch_size"]),
            finetune=warm_start,
        )

    def _pred_single_model(self, model, X_test, D_test, verbose, **kwargs):
        return model.predict(X_tab=X_test)

    def _data_preprocess(self, df, derived_data, model_name):
        # SettingWithCopyWarning in TabPreprocessor.transform
        # i.e. df_cont[self.standardize_cols] = self.scaler.transform(df_std.values)
        from pandas._config import option_context

        with option_context("mode.chained_assignment", None):
            X_df = self.tab_preprocessor.transform(df)
        return X_df, derived_data

    def _get_model_names(self):
        return [
            "TabMlp",
            "TabResnet",
            "TabTransformer",
            "TabNet",
            "SAINT",
            "ContextAttentionMLP",
            "SelfAttentionMLP",
            "FTTransformer",
            "TabPerceiver",
            "TabFastFormer",
        ]


class WideDeepCallback(Callback):
    def __init__(self, total_epoch, verbose):
        super(WideDeepCallback, self).__init__()
        self.val_ls = []
        self.total_epoch = total_epoch
        self.verbose = verbose

    def on_epoch_end(
        self,
        epoch: int,
        logs: Optional[Dict] = None,
        metric: Optional[float] = None,
    ):
        train_loss = logs["train_loss"]
        val_loss = logs["val_loss"]
        self.val_ls.append(val_loss)
        if epoch % src.setting["verbose_per_epoch"] == 0 and self.verbose:
            print(
                f"Epoch: {epoch + 1}/{self.total_epoch}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, "
                f"Min val loss: {np.min(self.val_ls):.4f}"
            )
