#!/usr/bin/env python
# coding: utf-8
# %%

# %%


"""Dataloaders."""


# %%


import os


# %%

import pickle
import numpy as np
import pandas as pd

from sklearn import preprocessing



# %%


IM_WIDTH = IM_HEIGHT = 160


# %%


def preprocess_adult(df):
  """Pre-process the Adult dataset.

  Args:
    df: pandas data frame.

  Returns:
  """
  df = df.dropna()

  # Here we apply discretisation on column marital_status
  df.replace(
      [
          'Divorced',
          'Married-AF-spouse',
          'Married-civ-spouse',
          'Married-spouse-absent',
          'Never-married',
          'Separated',
          'Widowed',
      ],
      [
          'not married',
          'married',
          'married',
          'married',
          'not married',
          'not married',
          'not married',
      ],
      inplace=True,
  )

  label_encoder = preprocessing.LabelEncoder()

  # Perform one-hot encoding on categorical features
  categorical_features = [
      'workclass',
      'education',
      'marital-status',
      'occupation',
      'relationship',
      'race',
      'gender',
      'native-country',
      'income',
  ]

  for feature in categorical_features:
    df[feature] = label_encoder.fit_transform(df[feature])

  # Split the dataset into features and target variable
  data = df.drop('income', axis=1)

  for n in data.columns:
    data[n] = (data[n] - data[n].mean()) / data[n].std()

  x = np.array(data.values, dtype=np.float32)

  y = np.array(df['income'], dtype=np.int32)
  a = np.array(df['gender'], dtype=np.int32)
  return x, y, a


# %%


def read_adult(path='../data/adult'):
  """Read the Adult dataset.

  Args:
    path:

  Returns:
  """

  columns = [
      'age',
      'workclass',
      'fnlwgt',
      'education',
      'education-num',
      'marital-status',
      'occupation',
      'relationship',
      'race',
      'gender',
      'capital gain',
      'capital loss',
      'hours per week',
      'native-country',
      'income',
  ]

  with open(os.path.join(path, 'adult.data'), 'rb') as f:
    train_df = pd.read_csv(f, names=columns)

  x_train, y_train, a_train = preprocess_adult(train_df)
  x_test, y_test, a_test = [], [], []

  return x_train, x_test, y_train, y_test, a_train, a_test


# %%


def preprocess_census(df):
  """Pre-process the Census dataset.

  Args:
    df:

  Returns:
  """
  df.dropna(inplace=True)

  categorical_features = [
      'class_worker',
      'education',
      'hs_college',
      'marital_stat',
      'major_ind_code',
      'major_occ_code',
      'race',
      'hisp_origin',
      'sex',
      'union_member',
      'unemp_reason',
      'full_or_part_emp',
      'tax_filer_stat',
      'region_prev_res',
      'state_prev_res',
      'det_hh_fam_stat',
      'det_hh_summ',
      'mig_chg_msa',
      'mig_chg_reg',
      'mig_move_reg',
      'mig_same',
      'mig_prev_sunbelt',
      'fam_under_18',
      'country_father',
      'country_mother',
      'country_self',
      'citizenship',
      'vet_question',
      'income_50k',
  ]
  for feature in categorical_features:
    label_encoder = preprocessing.LabelEncoder()
    df[feature] = label_encoder.fit_transform(df[feature])

  y = np.array(df['income_50k'], dtype=np.int32)
  a = np.array(df['sex'], dtype=np.int32)
  df.drop(columns=['income_50k'], inplace=True)
  df.drop(columns=['unk'], inplace=True)

  for n in df.columns:
    df[n] = (df[n] - df[n].mean()) / df[n].std()

  x = np.array(df.values, dtype=np.float32)
  return x, y, a


# %%


def read_census(path='../data/census/'):
  """Read the Census dataset. 
  
  Column names borrowed from: 
  https://docs.1010data.com/Tutorials/MachineLearningExamples/CensusIncomeDataSet.html
  1 unidentified column name marked as 'unk' and dropped later.
  
  Args:
    path:

  Returns:

  """
  column_names = [
      'age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education',
      'wage_per_hour', 'hs_college', 'marital_stat', 'major_ind_code',
      'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
      'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses',
      'stock_dividends', 'tax_filer_stat', 'region_prev_res', 'state_prev_res',
      'det_hh_fam_stat', 'det_hh_summ', 'unk', 'mig_chg_msa', 'mig_chg_reg',
      'mig_move_reg', 'mig_same', 'mig_prev_sunbelt', 'num_emp', 'fam_under_18',
      'country_father', 'country_mother', 'country_self', 'citizenship',
      'own_or_self', 'vet_question', 'vet_benefits', 'weeks_worked',
      'year', 'income_50k',
  ]

  # we only use the test set for online learning
  with open(os.path.join(path, 'census-income.data'), 'rb') as f:
    df = pd.read_csv(f, names=column_names)
  x, y, a = preprocess_census(df)
  return x, [], y, [], a, []


def read_jigsaw(path='../data/jigsaw/'):
  with open(os.path.join(path, "jigsaw.pkl"), "rb") as f:
    inputs, texts, Y, A = pickle.load(f)
  
  X = np.array(inputs)
  Y = np.array(Y)
  A = np.array(A)
  return X, Y, A


def read_compas(path="../data/compas"):

  feature_names = [
          "juv_fel_count",
          "juv_misd_count",
          "juv_other_count",
          "priors_count",
          "age",
          "c_charge_degree",
          "c_charge_desc",
          "age_cat",
          "sex",
          "race",
          "is_recid"]
  categorical_features = ["c_charge_degree",
          "c_charge_desc",
          "age_cat",
          "sex",
          "race",
          "is_recid"]
  train_df = pd.read_csv(os.path.join(path, "train.csv"), 
                         names=feature_names, header=None)
  test_df = pd.read_csv(os.path.join(path, "test.csv"), 
                        names=feature_names, header=None)


  df = pd.concat([train_df, test_df])
  df = df.dropna()

  mapping = {'White': 1, 'Black': 0, 'Other': 0}
  df['race'] = df['race'].map(mapping)

  label_encoder = preprocessing.LabelEncoder()
  for feature in categorical_features:
      df[feature] = label_encoder.fit_transform(df[feature])
  
  Y = np.array(df['is_recid'], dtype=np.float32)
  A = np.array(df['race'], dtype=np.float32)

  df = df.drop('is_recid', axis=1)
  X = np.array(df.values, dtype=np.float32)

  return X, Y, A


# CelebA

def load_dump(file_path):
  with open(file_path, "rb") as f:
    data = pickle.load(f)
  return data


def read_celeba(path="../data/celeba/"):
  X, Y, A = load_dump(os.path.join(path, "clip_data.pkl"))
  x_train, y_train, a_train = X, Y[: len(X)], A[: len(X)]
  return x_train, y_train, a_train
