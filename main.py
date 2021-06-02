import sqlite3
from sqlite3 import Error
from datetime import datetime as dt

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso


def load_otodom():
    """Load csv files with defined column names"""

    data_ads_cols = ["date", "user_id", "ad_id", "category_id", "params"]
    data_replies_cols = ["date", "user_id", "ad_id", "mails", "phones"]
    data_segmentation_cols = ["user_id", "segment"]
    data_categories_cols = ["category_id", "category_name"]

    # here you can find information about the announcements
    data_ads_df = pd.read_csv("data/data_ads.csv", delimiter=";", names=data_ads_cols)
    # information about the response per advertisement per day
    data_replies_df = pd.read_csv("data/data_replies.csv", delimiter=";", names=data_replies_cols)
    # segmentation mapping for each user
    data_segments_df = pd.read_csv("data/data_segments.csv", delimiter=";", names=data_segmentation_cols)
    # mapping to category tree
    data_categories_df = pd.read_csv("data/data_categories.csv", delimiter=";", names=data_categories_cols)

    return [data_ads_df, data_replies_df, data_segments_df, data_categories_df]


def cut_missing(source):
    """Cut rows with missing values from original source and make new df with only null values"""

    # list of null indicies
    null_indices = source[source["phones"].isnull()].index.tolist()

    # copping nulls
    null_list = []
    for i in null_indices:
        null_list.append(source.iloc[i])

    # dropping nulls
    not_null_replies = source.drop(null_indices)
    not_null_replies.reset_index(drop=True, inplace=True)

    # new DataFrame with missing values
    data_replies_cols = ["date", "user_id", "ad_id", "mails", "phones"]
    null_replies = pd.DataFrame(null_list, columns=data_replies_cols)
    null_replies.reset_index(drop=True, inplace=True)

    print("data_replies_df splitted into: null_replies, not_null_replies\n")

    return [null_replies, not_null_replies]


def check_missing(source, df_names):
    """Check percent of missing values in your columns"""

    for df, names in zip(source, df_names):
        print(f"Missing in {names} %\n", round(df.isnull().sum() / len(df) * 100, 2), "\n")


def select_split(source):
    """Select features and target, then split it to train test"""

    # features
    X = source.iloc[:, 0:4]
    # target
    y = source.iloc[:, 4]

    # split into train=0.8, test=0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    print("not_null_replies splitted into: X_train, X_test, y_train, y_test\n")

    return [X_train, X_test, y_train, y_test]


def best_pipe_model(X_train, X_test, y_train, y_test, to_pred):
    """Pipeline with grid search cv"""

    # feature's data
    to_pred = to_pred.drop(["phones"], axis=1)
    splits = [X_train, X_test, to_pred]

    # convert string data to datatime and, then convert to ordinal
    for split in splits:
        split["date"] = pd.to_datetime(split["date"])
        split["date"] = split["date"].apply(lambda x: x.toordinal())

    # pipeline
    pipe = Pipeline([("regressor", LinearRegression())])

    # search space with models and parameters
    search_space = [
        {"regressor": [LinearRegression()],
         "regressor__normalize": [False, True]},

        {"regressor": [Ridge()],
         "regressor__alpha": np.linspace(0, 0.5, 21),
         "regressor__max_iter": [1000, None]},

        {"regressor": [Lasso()],
         "regressor__alpha": np.linspace(0, 0.2, 21),
         "regressor__max_iter": [1000, None]}
    ]

    # standar scaling
    scaler = StandardScaler().fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)
    X_pred_std = scaler.transform(to_pred)

    # gridsearch, fit, predict
    gridsearch = GridSearchCV(pipe, search_space, cv=4, verbose=1, n_jobs=-1)
    best_model = gridsearch.fit(X_train_std, y_train)
    preds = best_model.predict(X_pred_std)

    # round to int
    rounded_preds = [round(elem) for elem in preds]

    # scores
    best_score = gridsearch.best_score_
    print("best_score: ", best_score)

    return rounded_preds


def join_replies(source_1, source_2, pred):
    """Insert predicted values into null replies and join it with not null replies"""

    replies_1 = source_1
    replies_2 = source_2
    replies_2["phones"] = pred

    # join DataFrames in axis=0
    data_replies = replies_1.append(replies_2, ignore_index=True)

    return data_replies


def dp_job():
    # List of OLX DataFrames
    data_ads_df, data_replies_df, data_segments_df, data_categories_df = load_otodom()
    # Returns: [data_ads_df, data_replies_df, data_segments_df, data_categories_df]

    olx_tables_data = [data_ads_df, data_replies_df, data_segments_df, data_categories_df]
    olx_tables_names = ["data_ads_df", "data_replies_df", "data_segments_df", "data_categories_df"]

    # Check percent of missing values 
    check_missing(source=olx_tables_data, df_names=olx_tables_names)
    # Returns: None    

    # Split original source into null/not null replies
    null_replies, not_null_replies = cut_missing(source=data_replies_df)
    # Returns: [null_replies, not_null_replies]

    # Try to load files from csv or make new
    try:
        new_data_replies = pd.read_csv("data/new_data_replies.csv")
    except Exception as e:
        print("Error has occurred: ", e, "\n")
    finally:
        # Select features, target and split it
        X_train, X_test, y_train, y_test = select_split(source=not_null_replies)
        # Retruns: [X_train, X_test, y_train, y_test]

        train_test_data = [X_train, X_test, y_train, y_test]
        train_test_names = ["X_train", "X_test", "y_train", "y_test"]

        # Check percent of missing values 
        check_missing(source=train_test_data, df_names=train_test_names)
        # Returns: None

        # Find best classification estimator and predict null values
        preds = best_pipe_model(X_train, X_test, y_train, y_test, to_pred=null_replies)
        # Returns: preds

        # Compound replies
        new_data_replies = join_replies(source_1=not_null_replies, source_2=null_replies, pred=preds)
        # Returns: data_replies

        # Update data_replies_df 
        data_replies_df = new_data_replies
        data_replies_df.to_csv("data/new_data_replies.csv")

    return [data_ads_df, data_replies_df, data_segments_df, data_categories_df]


def sqlite3_connect(db_file):
    """Establish connection with local database"""

    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print("Connected to sqlite3 ver: ", sqlite3.version)
    except Error as e:
        print(e)

    return conn


# In[90]:


def sqlite3_insert(source, tables, connection):
    """Insert data to the database from DataFrames"""

    # [data_ads_df, data_replies_df, data_segments_df, data_categories_df]
    indices = [0, 1, 2, 3]

    for i, tab in zip(indices, tables):
        source[i].to_sql(tab, connection, if_exists='replace', index=False)
    print("Data inserted. Tables successfully made")


def sqlite3_query_1(connection):
    """Send request to db and return response in DataFrame"""

    # Returns number of ads for user_id in last 7 days
    query = """
    select user_id, count(distinct ad_id)
    from replies
    where date <= '2019-04-30' and date >= '2019-04-24'
    group by user_id
    order by user_id;
    """

    response_df = pd.read_sql_query(query, connection)
    return response_df


def sqlite3_query_2(connection):
    """Send request to db and return response in DataFrame"""

    # Returns the number of answers by phone and email for a user for each day separately
    query = """
    select date, user_id, sum(phones), sum(mails)
    from replies
    where date <= '2019-04-30' and date >= '2019-04-24'
    group by date, user_id
    order by date, user_id;
    """

    response_df = pd.read_sql_query(query, connection)
    return response_df


def sqlite3_query_3(connection):
    """Send request to db and return response in DataFrame"""

    # Returns the total number of responses by phone and email for a given user within a 7-day period
    query = """
    select user_id, sum(phones), sum(mails)
    from replies
    where date <= '2019-04-30' and date >= '2019-04-24'
    group by user_id
    order by user_id;
    """

    response_df = pd.read_sql_query(query, connection)
    return response_df


def sql_job(source):
    database = r"otodom.db"
    table_list = ["ads", "replies", "segments", "categories"]

    # Connect to database
    conn = sqlite3_connect(database)
    # Retruns: conn

    if conn is not None:
        # Insert csv to sqlite3 database
        sqlite3_insert(source=source, tables=table_list, connection=conn)
        # Returns: None

        # Enquire db
        query_1 = sqlite3_query_1(connection=conn)
        query_2 = sqlite3_query_2(connection=conn)
        query_3 = sqlite3_query_3(connection=conn)
        # Returns: response_df
        print("SQL job is done.")
    else:
        print("Error! cannot create the database connection.")

    return [query_1, query_2, query_3]


def liquidity_per_user():
    """
    Liquidity will be understood as % of advertisements which have received 
    at least 1 response (by phone or e-mail) within a period of 7 days 
    (including day 0 - the day of adding an day of adding an ad)
    """

    pass


def full_data_analysis():
    """ 
    Jupyter/R Markdown preferred for analysis
    Scripts can be in separate files, or as part of a notebook depending on
    selected methods
    Please present your final results and most important conclusions in the 
    form of a presentation (e.g. Google slides)
    """

    pass


def question_1():
    """ 
    What differences do you see between the segments in terms of the data 
    you have available (including liquidity)?
    """

    pass


def question_2():
    """What do you think might influence higher or lower levels of liquidity?"""

    pass


def main():
    # Data processing job, datatime and null values
    data = dp_job()

    # SQL job, make db, insert data and send requests
    query_df1, query_df2, query_df2 = sql_job(source=data)


if __name__ == '__main__':
    main()
