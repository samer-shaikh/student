import pandas as pd
import pathlib
import sys
import yaml
from sklearn.model_selection import train_test_split



def read_dataset(path):
    data = pd.read_csv(path + 'Exam_Score_Prediction.csv')
    return data

# def save_dataset(data,save_path):
#     pathlib.Path(save_path).mkdir(parents=True,exist_ok=True)
#     data.to_csv(save_path + '/Exam_Score_Prediction_processed.csv',index = False)

def split_dataset(data,test_size,seed):
    train,test = train_test_split(data,test_size=test_size,random_state=seed)
    return train,test


def build_fearures(data):
    data.dropna(inplace=True)
    data.drop('student_id',axis=1,inplace=True)
    data = pd.get_dummies(data,drop_first=True)
    return data

def save_dataset(train,test,save_path):
    pathlib.Path(save_path).mkdir(parents=True,exist_ok=True)
    train.to_csv(save_path + '/train.csv',index = False)
    test.to_csv(save_path + '/test.csv',index = False)

def main():
    corr_dir = pathlib.Path(__file__)
    home_dir = corr_dir.parent.parent.parent
    param_file = home_dir.as_posix() + '/params.yaml'
    with open(param_file, "r") as f:
        params = yaml.safe_load(f)['make_dataset']

    input_path = sys.argv[1]
    data_path = home_dir.as_posix() + input_path
    output_path = home_dir.as_posix() + '/data/processed'

    data = read_dataset(data_path)
    data = build_fearures(data)
    train,test = split_dataset(data,params['test_split'],params['seed'])
    save_dataset(train,test,output_path)

if __name__ == '__main__':
    main()