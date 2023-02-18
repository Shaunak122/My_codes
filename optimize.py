import optuna

''' Hyperparameter Optimization Limits
max_iter	50	1000
max_depth	2	60
max_leaf_nodes	2	12
learning rate	0.001	0.5
'''

# %%
import pandas as pd
import cv2
from PIL import Image
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor, NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder
# from catboost import CatBoostRegressor, Pool
import plotly.express as px
from collections import defaultdict, OrderedDict
import json
import dacite
import sys
import pickle
import argparse
from pathlib import Path
import cfgparams
import data_utils
import feat_gen
import pdb
from parameters import save_parameters,opt_model
pad_value = 0
cfg_params = cfgparams.CfgParams().load()
logger = data_utils.get_logger("Optimize", cfg_params.log_fn)
from optuna.samplers import TPESampler

# %%
def train_model(x_trn, y_trn, trial):
    # model = RandomForestRegressor()
    gbt_max_depth = trial.suggest_int("gbt_max_depth", 2, 60, log=True)
    gbt_max_iter = trial.suggest_int("gbt_max_iter", 50, 1000, log=True)
    gbt_max_leaf_nodes = trial.suggest_int("gbt_max_leaf_nodes", 2, 12, log=True)
    gbt_learning_rate = trial.suggest_float("gbt_learning_rate", 0.001, 0.5, log=True)
    model = HistGradientBoostingRegressor(loss='absolute_error', learning_rate=gbt_learning_rate, max_iter=gbt_max_iter,
                                          max_depth=gbt_max_depth, max_leaf_nodes=gbt_max_leaf_nodes,l2_regularization=0.1)
    model.fit(x_trn, y_trn.squeeze())
    return model


# %%
def get_preds(x_tes, model):
    preds = model.predict(x_tes)
    return preds.squeeze()


# %%
def get_train_middle(cfg_params, target, y_trn):
    def_val = data_utils.get_default_val(cfg_params, target)
    if pd.api.types.is_numeric_dtype(y_trn):
        trn_middle = y_trn.median(skipna=False)
        if pd.isna(trn_middle):
            trn_middle = def_val
    else:
        trn_middle = y_trn.mode(dropna=False)
        if pd.isna(trn_middle).all():
            trn_middle = def_val
    logger.info(f"train middle: {trn_middle}")
    return trn_middle


# %%
def pad_list(list_of_lists, pad_value):
    max_length = max([len(l) for l in list_of_lists])
    return [list([pad_value] * (max_length - len(l))) + list(l) for l in list_of_lists]


# %%
def get_trained_model(x_trn, y_trn, trial):
    scaler_x = StandardScaler()

    if pd.api.types.is_numeric_dtype(y_trn):
        x_trn = x_trn[~y_trn.isna()]
        y_trn = y_trn[~y_trn.isna()]
        transform_y = StandardScaler()
        y_trn = y_trn.values.reshape(-1, 1)
    elif isinstance(y_trn.iloc[0], list):
        transform_y = StandardScaler()
        # y_trn = np.array([np.array(y) for y in y_trn])
        y_trn = np.array([np.array(y) for y in y_trn], dtype=object)
        y_trn = pad_list(y_trn, pad_value)
    else:
        transform_y = OneHotEncoder(sparse=False)
        y_trn = y_trn.values.reshape(-1, 1)
    if len(y_trn) > 1:
        y_trn = transform_y.fit_transform(y_trn)
        x_trn = scaler_x.fit_transform(x_trn)
    logger.info(f"train data - x: {x_trn.shape}, y: {y_trn.shape}")
    if len(y_trn) > 1 and y_trn.shape[1] == 1:
        model = train_model(x_trn, y_trn, trial)
    else:
        logger.info(f"Model not trained due to target being {y_trn.shape}")
        return {'model': None, 'transform_x': None, 'transform_y': None}

    return {'model': model, 'transform_x': scaler_x, 'transform_y': transform_y}


# %%
def eval_model(x_tes, y_tes, model_info):
    model = model_info['model']
    tfm_x = model_info['transform_x']
    tfm_y = model_info['transform_y']
    if model is None:
        pred = np.repeat(np.nan, len(x_tes))
        mae_pred = np.nan
    else:
        x_tes = tfm_x.transform(x_tes)
        pred = get_preds(x_tes, model)

        pred[np.isnan(pred)] = np.nanmean(pred)
        if len(pred.shape) > 1:
            pred = tfm_y.inverse_transform(pred).squeeze()
        else:
            pred = tfm_y.inverse_transform(pred.reshape(-1, 1)).squeeze()
        if len(pred.shape) > 1:
            y_tes = np.array([np.array(y) for y in y_tes])
            # if dims do not match then 100% error
            mae_pred = 1 if y_tes.shape != pred.shape else (y_tes != pred).sum() / len(pred)
        elif pd.api.types.is_numeric_dtype(pred):
            y_tes.fillna(0, inplace=True)
            # print("Starting debug print")
            # print(len(y_tes))
            # print(len(y_tes[~y_tes.isna()]))
            # print(len(pred))
            # print(len(pred[~y_tes.isna()]))
            # print("Ending debug print")
            mae_pred = mean_absolute_error(y_tes[~y_tes.isna()], pred[~y_tes.isna()])
        else:
            y_tes.fillna("", inplace=True)
            pred = pd.Series(pred).fillna("").values
            mae_pred = (y_tes != pred).sum() / len(pred)

    return {'pred': pred, 'targ': y_tes, 'mae_pred': mae_pred}


# %%
def get_eval_info(cfg_params, model_info, target, dftrn, dftes):
    x_tes = get_feats(cfg_params, dftes)
    y_tes = dftes[target] if target in dftes.columns else pd.Series(pd.NA, range(len(dftes)))
    y_trn = dftrn[target]
    logger.info(f"{target} test data - x: {x_tes.shape}, y: {y_tes.shape}")
    eval_info = eval_model(x_tes, y_tes, model_info)
    train_middle = model_info['train_middle']
    mae_mean = get_train_mae(cfg_params, target, train_middle, y_tes, y_trn)
    eval_info['mae_mean'] = mae_mean
    eval_info['train_middle'] = train_middle
    eval_info['file_id'] = dftes['file_id']
    eval_info['Photo_ID'] = dftes['Photo_ID']

    # eval_df = pd.DataFrame(eval_info)
    # eval_df['fldr'] = eval_df['file_id'].apply(lambda x: x.split('/')[0])
    # logger.info(eval_df.groupby('fldr').apply(lambda df: abs(df['targ']-df['pred']).mean()))
    logger.info(f"eval info for {target}: MAE Predicted: {eval_info['mae_pred']}, MAE Mean: {eval_info['mae_mean']}")
    # eval_info['pred_mean'] = np.repeat(train_middle, len(y_tes))
    return eval_info


# %%
def get_train_mae(cfg_params, target, trn_middle, y_tes, y_trn):
    preds = np.repeat(trn_middle, len(y_tes))
    if pd.api.types.is_numeric_dtype(y_trn):
        logger.info("Data is numeric")
        if y_tes.isna().sum() == 0:
            logger.info("Test data has no nan. mae calculated")
            mae_mean = mean_absolute_error(y_tes, preds)
        else:
            perc_nans = y_tes.isna().sum() / len(y_tes)
            logger.info(f"Test data has {perc_nans * 100}% nans.")
            # if perc_nans < 0.05:
            logger.info("Replacing nans with 0. mae calculated")
            def_val = data_utils.get_default_val(cfg_params, target)
            y_tes.fillna(def_val, inplace=True)
            mae_mean = mean_absolute_error(y_tes, preds)
            # else:
            #    logger.info("error rate calculated")
            #    mae_mean = 1-np.isclose(y_tes.values, preds, equal_nan=True).sum()/len(y_tes)
    else:
        if y_tes.isna().sum() != 0:
            def_val = data_utils.get_default_val(cfg_params, target)
            y_tes.fillna(def_val, inplace=True)
        logger.info("Data is non numeric. error rate calculated")
        mae_mean = (y_tes.values != preds).sum() / len(y_tes)
    return mae_mean


# %%
def get_feats(cfg_params, df):
    fldrs = df['project_name'].unique()
    feats = []
    hist_feats = feat_gen.HistFeats()
    for mode in ('full_image', 'detected_objects', 'detected_faces'):
        feats.append(hist_feats.get_hist_feats(fldrs, mode))
    feats.append(hist_feats.get_cam_feats(df, fldrs)[:, :2])
    x = np.hstack(feats)
    return np.nan_to_num(x)


# %%
def get_model(cfg_params, target, dftrn, trial):
    # if args.mode == 'train':
    x_trn = get_feats(cfg_params, dftrn)
    y_trn = dftrn[target]
    logger.info(f"{target} train data - x: {x_trn.shape}, y: {y_trn.shape}")
    model = get_trained_model(x_trn, y_trn, trial)
    model['train_middle'] = get_train_middle(cfg_params, target, y_trn)

    return model


def run(trial):
    np.random.seed(0)
    cfg_params = cfgparams.CfgParams().load()
    dftrn, dftes = data_utils.get_data(cfg_params)
    # logger.info("Training data counts:")
    # logger.info(f"{dftrn['project_name'].value_counts()}")
    # logger.info("Testing data counts:")
    # logger.info(f"{dftes['project_name'].value_counts()}")
    # df_human = data_utils.load_data(cfg_params.human_lrdata_path)
    # df_human = df_human[df_human['project_name'].isin(cfg_params.tes_fldrs)].copy().reset_index()
    # logger.info("Human data counts:")
    # logger.info(f"{df_human['project_name'].value_counts()}")

    targets = cfg_params.targets
    models_info = {}
    evals_info = {}

    for i, target in enumerate(targets):
        if target == args.target:
            logger.info(f"Parameter {i}: {target}")
            if target not in dftrn.columns:
                logger.error(f"{target} not found")
                continue
            models_info[target] = get_model(cfg_params, target, dftrn, trial)
            evals_info[target] = get_eval_info(cfg_params, models_info[target], target, dftrn, dftes)


    with open(Path(cfg_params.experiment_path) / (cfg_params.experiment_name + "_models_info_optimized.pkl"), 'wb') as f:
        pickle.dump(models_info, f)
    # with open(Path(cfg_params.experiment_path) / (cfg_params.experiment_name + "_evals_info.pkl"), 'wb') as f:
    #     pickle.dump(evals_info, f)
    #return models_info, evals_info
    return evals_info[args.target]['mae_pred']

# %%
if __name__ == '__main__':
    cfg_params = cfgparams.CfgParams().load()
    logger = data_utils.get_logger("Optimize", cfg_params.log_fn)
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", help="Specify which target needs to be optimized ",
                        choices=["Temperature", "Tint", "Exposure2012"],
                        default="Temperature")
    args = parser.parse_args()
    optim_params = ["Temperature", "Tint", "Exposure2012"]

    for i, param in enumerate(optim_params):
        args.target = param
        logger.info(f"Optimizing Parameter {i}: {param}")
        sampler = TPESampler(seed=0)
        study = optuna.create_study(direction="minimize",sampler=sampler)
        study.optimize(run, n_trials=100)
        print(study.best_trial)
        save_parameters(study.best_trial.params,param)
        logger.info(f"Best hyperparameters for target {param} " + str(study.best_trial.params))

    # exp_info = run(cfg_params, args)
    # for k, v in exp_info[1].items():
    #     logger.info(f"eval info for {k}: MAE Predicted: {v['mae_pred']}, MAE Mean: {v['mae_mean']}")


