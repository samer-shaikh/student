import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pathlib
import sys
from sklearn.svm import SVR
import yaml
import joblib



def train_model(x,y,kernel,gamma):
    model = SVR(kernel=kernel,gamma=gamma)
    model.fit(x,y)
    return model

def save_model(model,output_path):
    joblib.dump(model,output_path + '/model1_svr.joblib')


def main():

    corr_dir = pathlib.Path(__file__)
    home_dir = corr_dir.parent.parent.parent
    param_file = home_dir.as_posix() + '/params.yaml'
    with open(param_file, "r") as f:
        params = yaml.safe_load(f)['train_model']

    input_path = sys.argv[1]
    data_path = home_dir.as_posix() + input_path
    output_path = home_dir.as_posix() + '/models'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
 
    TARGET = params['target']
    train_features = pd.read_csv(data_path + '/train.csv')
    X = train_features.drop(TARGET, axis=1)
    y = train_features[TARGET]
    trained_model = train_model(X,y,params['kernel'],params['gamma'])
    save_model(train_features,output_path)

if __name__ == '__main__':
    main()