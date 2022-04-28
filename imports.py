from array import array
from cmath import nan
from tkinter import Y
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Metrics
from sklearn.metrics import roc_auc_score, r2_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, RocCurveDisplay, precision_recall_curve
import scikitplot
from plot_metric.functions import BinaryClassification

# Pre processors
from sklearn.utils import resample
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Balanceamento
from imblearn.combine import SMOTEENN

# Models
from sklearn.svm import SVC
from sklearn.neighbors import LocalOutlierFactor
## Modelos ensemble
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Model Selection + Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from imblearn.pipeline import Pipeline

# Bayesian hyperparameters tuning (optuna)
import optuna
from optuna.samplers import TPESampler

# Classes de criacao e remocao de features
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

import warnings
warnings.filterwarnings('ignore')

# Remocao de outliers com KNN
class RemoveOutliers(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def fit_resample(self, X, y=None):
        cols_skip = [
            col
            for col in X.select_dtypes(include=[np.number]).columns
            if len(X[col].value_counts().index) <= 40
        ]

        for col in X.drop(columns=cols_skip).select_dtypes(include=[np.number]).columns:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)

            IQR = q3 - q1

            inf = max(q1 - 1.5 * IQR, X[col].min())
            sup = min(q3 + 1.5 * IQR, X[col].max())

            X = X[(X[col] >= inf) & (X[col] <= sup)]
            y = y.loc[X.index]

        return X, y


# Criacao de novas features
class FeaturesCreator(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def fit_transform(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        X["CREDITO_RECEITA"] = X["AMT_CREDIT"] / X["AMT_INCOME_TOTAL"]
        X["ANUIDADE_RECEITA"] = X["AMT_ANNUITY"] / X["AMT_INCOME_TOTAL"]
        X["BENS_RECEITA"] = X["AMT_GOODS_PRICE"] / X["AMT_INCOME_TOTAL"]
        X["RECEITA_PESSOA"] = X["AMT_INCOME_TOTAL"] / X["CNT_FAM_MEMBERS"]

        # =======================================

        X["TEMPO_EMPREGADO_PERCENTUAL"] = np.clip(X["DAYS_EMPLOYED"] / X["DAYS_BIRTH"], 0, 1)

        # =======================================

        X["NUM_ADULTOS"] = X["CNT_FAM_MEMBERS"] - X["CNT_CHILDREN"]
        X['PERCENTUAL_CRIANCAS'] = X['CNT_CHILDREN'] / X['CNT_FAM_MEMBERS']
        X["RECEITA_ADULTO"] = X["AMT_INCOME_TOTAL"] / X["NUM_ADULTOS"]

        # =======================================

        X['QTD_PAGAMENTOS'] = X['AMT_CREDIT'] / X['AMT_ANNUITY']

        # =======================================

        X["EXT_SOURCE_MIN"]  = X[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].min(axis = 1)
        X["EXT_SOURCE_MAX"]  = X[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].max(axis = 1)
        X["EXT_SOURCE_MEAN"] = X[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis = 1)
        X["EXT_SOURCE_STD"]   = X[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].std(axis = 1)
        X["NUM_EXT_SOURCES"] = (~X[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].isnull()).astype(int).sum(axis=1)

        # =======================================

        flag_docs = [x for x in X.columns if "FLAG_DOCUMENT_" in x]
        X["NUM_DOCUMENTOS"] = X[flag_docs].sum(axis = 1)
        
        X['FLAG_SUM'] = X[['FLAG_MOBIL',
                           'FLAG_EMP_PHONE',
                           'FLAG_WORK_PHONE',
                           'FLAG_CONT_MOBILE',
                           'FLAG_PHONE',
                           'FLAG_EMAIL']].sum(axis=1)

        # =======================================

        X["DAY_APPR_PROCESS_START"] = X["WEEKDAY_APPR_PROCESS_START"].apply(lambda x: "Weekend" if x in ["SATURDAY", "SUNDAY"] else "Working day")

        # =======================================

        X["OWN_CAR_AGE_RATIO"] = X["OWN_CAR_AGE"] / X["DAYS_BIRTH"]
        X["DAYS_ID_PUBLISHED_RATIO"] = X["DAYS_ID_PUBLISH"] / X["DAYS_BIRTH"]
        X["DAYS_REGISTRATION_RATIO"] = X["DAYS_REGISTRATION"] / X["DAYS_BIRTH"]
        X['DAYS_WITHOUT_REGISTRATION'] = X["DAYS_REGISTRATION"] - X['DAYS_EMPLOYED']
        
        # =======================================
        
        X['FOI_CASADO'] = X['NAME_FAMILY_STATUS'].apply(lambda x: 1 if x not in ['Single / not married', 'Unknown', np.nan] else 0)
        
        # =======================================
        
        X['REG_LIVE_REGION_CITY_SUM'] = X[['REG_REGION_NOT_LIVE_REGION',
                                           'REG_REGION_NOT_WORK_REGION',
                                           'LIVE_REGION_NOT_WORK_REGION',
                                           'REG_CITY_NOT_LIVE_CITY',
                                           'REG_CITY_NOT_WORK_CITY',
                                           'LIVE_CITY_NOT_WORK_CITY']].sum(axis=1)
        
        # =======================================
        
        X['AMT_REQ_CREDIT_BUREAU_SUM'] = X[['AMT_REQ_CREDIT_BUREAU_HOUR',
                                            'AMT_REQ_CREDIT_BUREAU_DAY',
                                            'AMT_REQ_CREDIT_BUREAU_WEEK',
                                            'AMT_REQ_CREDIT_BUREAU_MON',
                                            'AMT_REQ_CREDIT_BUREAU_QRT',
                                            'AMT_REQ_CREDIT_BUREAU_YEAR']].sum(axis=1)
        
        # =======================================
        
        X['REGION_RATING_DIFF'] = X['REGION_RATING_CLIENT'] - X['REGION_RATING_CLIENT_W_CITY']
        X['REGION_RATING_SUM'] = X['REGION_RATING_CLIENT'] + X['REGION_RATING_CLIENT_W_CITY']
        
        # =======================================
        
        X['HIGHER_EDUCATION'] = X['NAME_EDUCATION_TYPE'].apply(lambda x: 1 if x in ['Higher education', 'Academic degree'] else 0)
        
        # =======================================
        
        def definir_periodo(x):
            if x == np.nan:
                return x

            if x >= 0 and x < 6:
                return 'madrugada'
            elif x < 13:
                return 'manha'
            elif x < 19:
                return 'tarde'
            else:
                return 'noite'
        
        X['HOUR_APPR_PROCESS_START_PERIOD'] = X['HOUR_APPR_PROCESS_START'].apply(definir_periodo)
        
        return X


# Remocao de colunas
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, column_to_drop):
        self.column_to_drop = column_to_drop
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.drop(columns= self.column_to_drop)


# Renomear features
class RenameFeature(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, X_train, X_test, y=None):
        colunas = pd.get_dummies(X).columns
        
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
        
        X_train.columns = colunas
        X_test.columns = colunas
        
        return X_train, X_test
 
 
# Train-Test-Split
def tts(df, target='None', test_size=0.2, stratify=True, seed=42):
    X = df.drop(columns='TARGET')
    y = df[target]
    
    stratify = y if stratify else None
    
    return train_test_split(X, y, test_size=test_size, stratify=stratify, random_state=seed)


# Pre-processando os dados (generico)
def pre_processar(df_train:pd.DataFrame, df_test:pd.DataFrame, target:str, cols:dict, stratify:bool=True, features_test:str='median', cols_to_drop:array='None',
                  undersample:bool=False, oversample:bool=False, smoteenn:bool=False, sampling_strategy:float='auto', rem_outliers:bool=False, seed:int=42):
    '''
    df (DataFrame): DataFrame que deseja que seja feito os tratamentos de dados.
    target (str): Nome do target que esta no DF.
    cols (dict): O tratamento desejado para  cada coluna com dado faltante.
    stratify (bool): No train-test-split, se True, faz a estratificacao, se False, nao estratifica.
    features_test (str): Pode ser 'negative' ou 'median' para acrescentar no pre-processamento.
    cols_to_drop (array): Colunas que deseja remover.
    smoteenn (bool): Se True, habilita o pre-processamento SMOTEENN para balanceamento de features, se False, pula essa etapa.
    undersample (bool): Se True, faz o undersampling dos dados e automaticamente deabilita o 'smoteenn'.
    sampling_strategy (float): Se undersample ou oversample = True, faz o balanceio percentual em relacao a classe majoritaria.
    seed (int): A seed do random_state definido para os tramentos que necessitam dele.
    rem_outliers (bool): Se True, faz a remocao de outlires, se False, nao remove outliers.
    
    Retona: pre-processador, X_train, X_test, y_train, y_test
    '''
    
    X_train, X_test, y_train, y_test = tts(df_train, target=target, seed=seed, stratify=stratify)

    
    
    # Balanceamento de classe
    if undersample:
        smoteenn = oversample = False
        X_train, y_train = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=seed).fit_resample(X_train, y_train)
    if oversample:
        smoteenn = False
        X_train, y_train = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=seed).fit_resample(X_train, y_train)

    # Aplicando criacao e remocao de features
    pre_process = Pipeline([("create_feature", FeaturesCreator()),
                            ("column_dropper", ColumnDropper(cols_to_drop))])

    pre_process.fit(X_train)
    X_train = pre_process.transform(X_train)
    X_test = pre_process.transform(X_test)
    X_val = pre_process.transform(df_test)

    # Features numericas
    median_pipe = Pipeline([('impute_median', SimpleImputer(strategy='median')),
                            ('numeric_std', StandardScaler())])
    features_median = cols['col_median']

    negative_pipe = Pipeline([('impute_negative', SimpleImputer(strategy='constant', fill_value=-1)),
                              ('numeric_std', StandardScaler())])
    features_negative = cols['col_neg_val']

    zero_pipe = Pipeline([('impute_zero', SimpleImputer(strategy='constant', fill_value=0)),
                          ('numeric_std', StandardScaler())])
    features_zero = cols['col_zero']
    
    # Se surgir novas colunas numericas faltando:
    numeric_pipe = Pipeline([('num_imput',  SimpleImputer(strategy='median')),
                             ('numeric_std', StandardScaler())])
    features_num = X_train.select_dtypes(include=np.number).columns

    if features_test == 'negative':
        features_negative.extend(cols['col_neg_or_median'])
    elif features_test == 'median':
        features_median.extend(cols['col_neg_or_median'])
    
    # Features categoricas (nao numericas)
    cat_pipe = Pipeline([('impute_cat', SimpleImputer(strategy='constant', fill_value='unknown')),
                         ('onehot', OneHotEncoder(handle_unknown='ignore', categories='auto'))])
    features_cat = cols['col_unknown']
    
    # Se surgir novas colunas numericas faltando:
    cat_constant_pipe = Pipeline([('cat_constant', SimpleImputer(strategy='most_frequent')),
                                  ('onehot', OneHotEncoder(handle_unknown='ignore', categories='auto'))])
    features_cat_constant_rest = X_train.select_dtypes(exclude=np.number).columns

    # Criando e unificando o pre-processador em um objeto
    pre_processor = ColumnTransformer([('median', median_pipe, features_median),
                                       ('negative', negative_pipe, features_negative),
                                       ('zero', zero_pipe, features_zero),
                                       ('numerical_rest', numeric_pipe, features_num),
                                       ('categorical_rest', cat_constant_pipe, features_cat_constant_rest),
                                       ('categorical', cat_pipe, features_cat)])
    
    if rem_outliers:
        X_train, y_train = RemoveOutliers().fit_resample(X_train, y_train)
    
    if smoteenn:
        pre_processor = Pipeline([('pp', pre_processor),
                                  ('smoteenn', SMOTEENN(random_state=seed, n_jobs=-1, sampling_strategy=sampling_strategy))])
    
    return pre_processor, X_train, X_test, X_val, y_train, y_test


# Mostrando resultados ROC-AUC
def pontuacao_roc_auc(modelo, X_train, X_test, y_train, y_test):
    '''
    modelo: modelo utilizado para obter as predicoes
    X_train: X de treino
    X_test: X de teste
    y_train: y de treino
    y_test: y de teste

    Mostra a matrix de confusao dos dados de treino e teste;
    Mostra a curva ROC-AUC;
    Printa o score roc-auc dos dados de teste.
    '''
    y_train_proba = modelo.predict_proba(X_train)
    y_test_proba = modelo.predict_proba(X_test)
    
    y_train_proba_1 = modelo.predict_proba(X_train)[:, 1]
    y_test_proba_1 = modelo.predict_proba(X_test)[:, 1]
    
    y_train_pred = modelo.predict(X_train)
    y_test_pred = modelo.predict(X_test)
    

    # Plotando Confusion matrix train
    print("\nMatriz de confusão do modelo nos dados de TREINO:")
    print(confusion_matrix(y_train, y_train_pred))

    ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, cmap="viridis")
    plt.show()

    print("\nClassification report do modelo nos dados de treino:")
    print(classification_report(y_train, y_train_pred))
    
    roc_auc = RocCurveDisplay.from_predictions(y_train, y_train_proba_1);
    roc_auc.ax_.set_title(f"AUC: {roc_auc_score(y_train, y_train_proba_1):.3f}", fontsize=16)
    x = np.linspace(0, 1, 100)
    plt.plot(x, x, ls=":", color="black")
    plt.show()

    print(f'\033[33mROC-AUC score TREINO:\033[0m {roc_auc_score(y_train, y_train_proba_1)}')
    scikitplot.metrics.plot_roc_curve(y_train, y_train_proba);
    plt.show();
    bc = BinaryClassification(y_train, y_train_proba_1, labels=["Class 1"])

    # Figures
    plt.figure(figsize=(5,5));
    bc.plot_roc_curve();
    plt.show();

    print("\n##########################################################\n")

    # Plotando Confusion matrix test
    print("\nMatriz de confusão do modelo nos dados de TESTE:")
    print(confusion_matrix(y_test, y_test_pred))

    ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, cmap="viridis");
    plt.show();

    print("\nClassification report do modelo nos dados de teste:")
    print(classification_report(y_test, y_test_pred))

    # Plotando a curva ROC-AUC dos dados de teste
    roc_auc = RocCurveDisplay.from_predictions(y_test, y_test_proba_1);
    roc_auc.ax_.set_title(f"AUC: {roc_auc_score(y_test, y_test_proba_1):.3f}", fontsize=16)
    x = np.linspace(0, 1, 100)
    plt.plot(x, x, ls=":", color="black")
    plt.show()

    print(f'\033[32mROC-AUC score TEST: {roc_auc_score(y_test, y_test_proba_1)}\033[0m')
    scikitplot.metrics.plot_roc_curve(y_test, y_test_proba);
    plt.show();
    
    bc = BinaryClassification(y_test, y_test_proba_1, labels=["Class 1"])

    # Figures
    plt.figure(figsize=(5,5));
    bc.plot_roc_curve();
    plt.show();


# Removendo outliers
def remove_outliers(df, colunas_num_categoricas):
    for col in df.drop(columns='TARGET').select_dtypes(include=[np.number]).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        
        IQR = q3 - q1
        
        inf = max(q1 - 1.5 * IQR, df[col].min())
        sup = min(q3 + 1.5 * IQR, df[col].max())
        
        df_out = df[(df[col] >= inf) & (df[col] <= sup)]
    return df_out

    
def show_balance(y):
    print(pd.concat((pd.DataFrame(y.value_counts()), pd.DataFrame(y.value_counts()/y.shape[0])), axis=1))
    sns.countplot(y);