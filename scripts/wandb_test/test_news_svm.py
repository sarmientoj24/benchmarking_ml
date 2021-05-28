from utils import TextCleaner
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.svm import SVC
import argparse
import os
import wandb
from sklearn.metrics import classification_report

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj", help="wandb project",
                        type=str)
    parser.add_argument("--name", help="wandb experiment name",
                        type=str)
    parser.add_argument("--entity", help="wandb entity name",
                        type=str)
    
    args = parser.parse_args()

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
            max_features=5000))
    ], verbose=True)

    model = SVC(kernel='linear')

    # Init Wandb connection
    wandb.init(
        project=args.proj, 
        entity=args.entity, 
        name=args.name
    )

    X_train_trans = pipeline.fit_transform(X_train)
    print("Start model fitting...")
    model.fit(X_train_trans, y_train)

    print("Start model evaluation")
    processed_input = pipeline.transform(X_test)
    prediction = model.predict(processed_input)
    print(classification_report(y_test, prediction))


if __name__ == '__main__':
    main()