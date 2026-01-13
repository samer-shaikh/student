import pandas as pd
import pathlib
import boto3
import io


def read_dataset():
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket="students-mlops",Key="Exam_Score_Prediction.csv")
    file = response["Body"].read()
    data = pd.read_csv(io.BytesIO(file),sep=',')
    return data

def save_dataset(data,save_path):
    pathlib.Path(save_path).mkdir(parents=True,exist_ok=True)
    data.to_csv(save_path + '/Exam_Score_Prediction_processed.csv',index = False)

def build_fearures(data):
    data.dropna(inplace=True)
    data.drop('student_id',axis=1,inplace=True)
    data = pd.get_dummies(data,drop_first=True)
    return data

def main():
    corr_dir = pathlib.Path(__file__)
    home_dir = corr_dir.parent.parent.parent
    output_path = home_dir.as_posix() + '/data/raw'

    data = read_dataset()
    data = build_fearures(data)
    save_dataset(data,output_path)

if __name__ == '__main__':
    main()