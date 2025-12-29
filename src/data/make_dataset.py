import pandas as pd
import pathlib
import sys
import yaml
from sklearn.model_selection import train_test_split


def read_dataset(path):
    data = pd.read_csv(path)
    return data

def split_dataset(data,test_size,seed):
    train,test = train_test_split(data,test_size=test_size,random_state=seed)
    return train,test

def save_dataset(train,test,save_path):
    pathlib.Path(save_path).mkdir(parents=True,exist_ok=True)
    train.to_csv(save_path + '/train.csv',index = False)
    test.to_csv(save_path + '/test.csv',index = False)

def main():
    corr_dir = pathlib.Path(__file__)
    home_dir = corr_dir.parent.parent.parent
    param_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load((param_file))['make_data']

    input_path = sys.argv[1]
    data_path = home_dir.as_posix() + input_path
    output_path = home_dir.as_posix() + '/data/processed'

    data = read_dataset(data_path)
    train,test = split_dataset(data,params['test_size'],params['seed'])
    save_dataset(train,test,output_path)

if __name__ == '__main__':
    main()