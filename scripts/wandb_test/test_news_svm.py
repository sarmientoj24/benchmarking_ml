from utils import TextCleaner
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.svm import SVC
import argparse
import os
import wandb
from sklearn.metrics import classification_report
import time
from timerit import Timerit

def main():
    # Parse arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--proj", help="wandb project",
    #                     type=str)
    # parser.add_argument("--name", help="wandb experiment name",
    #                     type=str)
    # parser.add_argument("--entity", help="wandb entity name",
    #                     type=str)
    
    # args = parser.parse_args()

    # define training and test data directories
    data_dir = 'dataset/AGNEWS/'
    train_file = os.path.join(data_dir, 'train.csv')
    test_file = os.path.join(data_dir, 'test.csv')

    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    X_train = train_data['text'].to_list()[:20000]
    y_train = train_data['label'].to_list()[:20000]

    X_test = test_data['text'].to_list()[:1000]
    y_test = test_data['label'].to_list()[:1000]

    pipeline = Pipeline([
        ('cleaner', TextCleaner()),
        ('tfidf',  TfidfVectorizer(
            use_idf=True,
            min_df=5,
            max_df=0.95,
            lowercase=True,
            ngram_range=(1, 2),
            max_features=1000))
    ], verbose=True)

    model = SVC(kernel='linear', max_iter=100)

    # Init Wandb connection
    # wandb.init(
    #     project=args.proj, 
    #     entity=args.entity, 
    #     name=args.name
    # )

    full_duration_s = time.time()
    only_train_duration_s = time.time()

    ### PIPELINE FIT
    pipeline_fit_s = time.time()
    X_train_trans = pipeline.fit_transform(X_train)
    pipeline_fit_e = time.time()
    print("[TRAIN] Pipeline Fit Time: ", pipeline_fit_e - pipeline_fit_s)

    model_fit_s = time.time()
    print("[TRAIN] Start model fitting...")
    model.fit(X_train_trans, y_train)
    model_fit_e = time.time()
    print("[TRAIN] FIT Time: ", model_fit_e - model_fit_s)

    only_train_duration_e = time.time()
    print("[TRAIN] FULL TRAIN duration: ", only_train_duration_e - only_train_duration_s)

    only_test_duration_s = time.time()

    pipeline_fit_s = time.time()
    print("[TEST] Start model evaluation...")
    processed_input = pipeline.transform(X_test)
    pipeline_fit_e = time.time()
    print("[TEST] PIPELINE Fit Time: ", pipeline_fit_e - pipeline_fit_s)

    model_fit_s = time.time()
    print("[TEST] Start model evaluation...")
    prediction = model.predict(processed_input)
    model_fit_e = time.time()
    print("[TEST] PREDICT Time: ", model_fit_e - model_fit_s)

    only_test_duration_e = time.time()
    print("[TEST] FULL TEST duration: ", only_test_duration_e - only_test_duration_s)
    
    
    full_duration_e = time.time()
    print("[ALL] Full Duration Time: ", full_duration_e - full_duration_s)

    print(classification_report(y_test, prediction))


if __name__ == '__main__':
    main()