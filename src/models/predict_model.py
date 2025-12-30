from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from dvclive import Live
import joblib
import pathlib
import pandas as pd
import sys
import yaml

def load_model(path):
    model = joblib.load(path)

def pridect_model(model,data,target):

    X = data.drop(target, axis=1)
    y = data[target]

    y_pred = model.pridect(X)
    return y, y_pred



def main():
    corr_dir = pathlib.Path(__file__)
    home_dir = corr_dir.parent.parent.parent
    model_path = home_dir.as_posix() + '/model/svr.joblib'
    param_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load((param_file))['train_model']

    input_path = sys.argv[1]
    data_path = home_dir.as_posix() + input_path

    model = load_model(model_path)
    test_features = pd.read_csv(data_path + '/test.csv')
    TARGET = params['target']

    y_test, y_pred = pridect_model(model,test_features,TARGET)

    

    with Live(save_dvc_exp=True) as live:
        live.log_metric('train/r2_score',r2_score(y_test,y_pred))
        live.log_metric('train/MSE',mean_squared_error(y_test,y_pred))
        live.log_metric('train/MAE',mean_absolute_error(y_test,y_pred))
        live.next_step()
