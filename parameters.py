import cfgparams
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
cfg_params = cfgparams.CfgParams().load()
def save_parameters(hyp_dict,param):
      df=pd.DataFrame(hyp_dict.items(),columns=['parameters','values'])
      df.to_csv(cfg_params.experiment_path+f'/{param}_param.csv')



def get_values(df):
    df=df.set_index('parameters')
    gbt_max_depth= df.loc['gbt_max_depth']['values']
    gbt_max_iter = df.loc['gbt_max_iter']['values']
    gbt_max_leaf_nodes	= df.loc['gbt_max_leaf_nodes']['values']
    gbt_learning_rate	=df.loc['gbt_learning_rate']['values']
    return gbt_max_depth,int(gbt_max_iter),gbt_max_leaf_nodes,gbt_learning_rate

def opt_model(param):
  try:
    if param=="Temperature":
        df=pd.read_csv(cfg_params.experiment_path+f'/{param}_param.csv')
        model=HistGradientBoostingRegressor(loss='absolute_error',max_depth=get_values(df)[0],max_iter=get_values(df)[1],max_leaf_nodes=get_values(df)[2],learning_rate=get_values(df)[3])
        return model
    elif param=='Tint':
        df=pd.read_csv(cfg_params.experiment_path+f'/{param}_param.csv')
        model=HistGradientBoostingRegressor(loss='absolute_error',max_depth=get_values(df)[0],max_iter=get_values(df)[1],max_leaf_nodes=get_values(df)[2],learning_rate=get_values(df)[3])
        return model
    elif param=="Exposure2012":
        df=pd.read_csv(cfg_params.experiment_path+f'/{param}_param.csv')
        model=HistGradientBoostingRegressor(loss='absolute_error',max_depth=get_values(df)[0],max_iter=get_values(df)[1],max_leaf_nodes=get_values(df)[2],learning_rate=get_values(df)[3])
        return model
  except:
      model = HistGradientBoostingRegressor(loss='absolute_error', learning_rate=0.01, max_iter=1000)
      return model

if __name__ == '__main__':
    cfg_params = cfgparams.CfgParams().load()
