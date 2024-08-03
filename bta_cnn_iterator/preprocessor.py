import os, re
import pandas as pd

def create_input_from_text_gold(df):
    df["PREFIX"] = df.apply(lambda row: row["SCHEMA_NAME"] + "." + row["TABLE_NAME"], axis=1)
    col_df = df[["PREFIX", "COLUMN_NAME"]]

    col_df = col_df.groupby("PREFIX").agg(list).reset_index()
    col_df["COLUMN_NAME"] = col_df["COLUMN_NAME"].apply(lambda x: ", ".join(x))
    col_df["REL_TABLE_NAME"] = col_df.apply(lambda row: row["PREFIX"] + "(" + str(row["COLUMN_NAME"] + ")"), axis=1)
    col_df = col_df[["REL_TABLE_NAME"]]
    bt_df = df[~df["BUSINESS_TERM"].isna()][["BUSINESS_TERM"]].drop_duplicates()
    return col_df, bt_df


def create_col_table(tab_name):
    col_file_name = tab_name + "_Columns_Relation.txt"
    col_file_path = os.path.join("sampledata", col_file_name)
    cols = open(col_file_path, "r")
    cols = "".join([i for i in cols])
    cols = re.split("[\n]", cols)[1:-1]
    cols = [re.split("[.(), ]", i) for i in cols]
    cols = [list(filter(None, i)) for i in cols]
    cols = [i[0:2] + [i[2:]] for i in cols]
    col_df = pd.DataFrame(cols, columns=["SCHEMA_NAME", "TABLE_NAME", "COLUMN_NAME"])
    col_df = col_df.explode("COLUMN_NAME").reset_index()
    col_df["index"] = col_df.index
    col_df = col_df.rename({"index": "col_id"}, axis=1)
    return col_df


def create_bt_table(tab_name):
    bt_file_name = tab_name + "_BT.txt"
    bt_file_path = os.path.join("sampledata", bt_file_name)
    bt = open(bt_file_path, "r")
    bt = "".join([i for i in bt])
    bt = re.split("[\n]", bt)[1:-1]
    bt_df = pd.DataFrame(bt, columns=["BUSINESS_TERM"]).drop_duplicates().reset_index()
    bt_df["index"] = bt_df.index
    bt_df = bt_df.rename({"index": "bt_id"}, axis=1)
    return bt_df
