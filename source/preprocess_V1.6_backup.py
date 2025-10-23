import sys
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from feature_engine import imputation as mdi
from feature_engine import encoding as ce
from feature_engine.discretisation import EqualFrequencyDiscretiser
from feature_engine.discretisation import EqualWidthDiscretiser
from feature_engine.encoding import OneHotEncoder
from dateutil.parser import parse
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
#from feature_engine.imputation import MeanMedianImputer
#from feature_engine.encoding import OneHotEncoder, OrdinalEncoder
#from feature_engine.discretisation import EqualFrequencyDiscretiser, EqualWidthDiscretiser
from sklearn.base import BaseEstimator, TransformerMixin # CustomOrdinalEncoder를 위해 추가
from typing import List, Union # CustomOrdinalEncoder를 위해 추가

import mean_median2 as mm
import temporal_feature as tf
from pathlib import Path
import warnings
import traceback
import json

warnings.filterwarnings('ignore')


def join_abs_path(p1, p2):
    return os.path.abspath(os.path.join(p1, p2))
class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables: Union[None, List[str]] = None,
                 encoding_method: str = 'ordered',
                 suffix: str = '_ordinal'):

        if variables is not None and not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        self.encoding_method = encoding_method
        self.suffix = suffix
        self.encoder_ = None # feature-engine의 OrdinalEncoder 인스턴스
        self.feature_names_in_ = None # sklearn 스타일 입력 특성 이름

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        if self.variables is None:
            raise ValueError("Parameter 'variables' must be provided as a list of column names.")

        self.variables_ = self.variables # 실제로 변환될 변수들
        self.feature_names_in_ = X.columns.tolist()

        self.encoder_ = ce.OrdinalEncoder(
            encoding_method=self.encoding_method,
            variables=self.variables_
        )
        self.encoder_.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_input_for_internal_encoder = X.copy()

        # encoder_가 variables_에 지정된 컬럼들만 변환한 DataFrame을 반환
        X_processed_by_internal_encoder = self.encoder_.transform(X_input_for_internal_encoder)
        X_output = X.copy()

        for var in self.variables_:
            new_col_name = f"{var}{self.suffix}"
            X_output[new_col_name] = X_processed_by_internal_encoder[var] # 새 컬럼으로 추가 (DataFrame 끝에 붙음)

        X_output = X_output.drop(columns=self.variables_)
        return X_output

    def get_feature_names_out(self, input_features: List[str] = None) -> List[str]:
        if input_features is None:
            input_features = self.feature_names_in_
        output_features = [col for col in input_features if col not in self.variables_]
        for var in self.variables_:
            output_features.append(f"{var}{self.suffix}")
        return output_features

def position_Y_COL(cols):  # Y label을 가장 뒤로 위치 변경
    if Y_COL in cols:  # Y_COL이 있을 때만 remove 실행
        cols_copy = cols.copy()
        cols_copy.remove(Y_COL)
        return cols_copy + [Y_COL]
    else:  # Y_COL이 없으면 변경없이 리턴
        return cols

# 안전한 날짜 파싱 함수
def safe_parse(date_string):
    try:
        return parse(date_string)
    except (ValueError, TypeError):
        return None  # 또는 다른 오류 처리 방법

def read_data(afile):    
    #날짜컬럼은 temporal_feature.py에서 처리함
    # # Case 1: 날짜 컬럼이 없는 경우
    # if config_dict['date_col'] is np.nan:
    #     df = pd.read_csv(afile, usecols=config_dict['keep_col'])
    # # Case 2: 날짜 컬럼이 있는 경우
    # else:
    #     #date_col이 여러개면 안돌아갈 수 있음
    #     # # Case 2-1: date_col이 keep_col에 포함된 경우
    #     # if config_dict['date_col'] in config_dict['keep_col']: 

    #     # Case 2-1: date_col 중 하나라도 keep_col에 포함된 경우
    #     if any(col in config_dict['keep_col'] for col in config_dict['date_col']):
    #         df = pd.read_csv(afile, usecols=config_dict['keep_col'], parse_dates=config_dict['date_col'])
    #         # date_col이 정말 date형인지 확인
    #         for col in config_dict['date_col']:
    #             if not pd.api.types.is_datetime64_any_dtype(df[col]):
    #                 df[col] = df[col].apply(safe_parse)

    #     # Case 2-2: date_col이 keep_col에 없는 경우
    #     else:
    #         df = pd.read_csv(afile, usecols=config_dict['keep_col'])
    
    df = pd.read_csv(afile, usecols=config_dict['keep_col'],encoding='utf-8')
    cols = list(df.columns)
    cols = position_Y_COL(cols)
    return df[cols]  


def y_label_enc(df):
    df = df.copy()
    # 타겟 변수(Y_COL)에 결측치가 있는지 확인
    if df[Y_COL].isnull().any():
        Y_null_exist = True
    else:
        Y_null_exist = False
    labeler = LabelEncoder()
    # 타겟 변수를 숫자로 인코딩
    df[Y_COL] = labeler.fit_transform(df[Y_COL])
    return df, Y_null_exist

#분류별 컬럼 분류(discrete:셀수있음, continuous:연속형, categorical:오브젝트, 그외 날짜 데이터)
def discrete_cont(df):
    # 원본 데이터 보존을 위해 카피하여 작업함
    data = df.copy()
    # 날짜형, 시간형
    date_cols_len = len(config_dict['date_col']) if config_dict['date_col'] and not pd.isna(config_dict['date_col'][0]) else 0
    
    # json형
    dict_cols_len = len(config_dict['dict_col']) if config_dict['dict_col'] and not pd.isna(config_dict['dict_col'][0]) else 0
    # 벡터형
    vector_cols_len = len(config_dict['vector_col']) if config_dict['vector_col'] and not pd.isna(config_dict['vector_col'][0]) else 0
    # 진법형
    non_dec_cols_len = len(config_dict['non_dec_col']) if config_dict['non_dec_col'] and not pd.isna(config_dict['non_dec_col'][0]) else 0
    # 문장형
    sentence_cols_len = len(config_dict['sentence_col']) if config_dict['sentence_col'] and not pd.isna(config_dict['sentence_col'][0]) else 0

    # Case 1 : 날짜 컬럼이 없으면
    if date_cols_len < 1:
        # 이산형 변수: 숫자형이면서 고유값이 임계값보다 적은 경우
        discrete = [var for var in data.columns if
                    data[var].dtype != 'O' and var != Y_COL and data[var].nunique() < config_dict['discrete_thresh_hold']]
        # 연속형 변수: 숫자형이면서 이산형이 아닌 경우
        continuous = [var for var in data.columns if
                      data[var].dtype != 'O' and var != Y_COL and var not in discrete]
    # Case 2 : 날짜 컬럼이 있으면
    else:
        # 이산형 변수: 숫자형이면서 고유값이 임계값보다 적은 경우 및 날자컬럼이 아닌 경우
        discrete = [var for var in data.columns if
                    data[var].dtype != 'O' and var != Y_COL and var not in config_dict['date_col']
                    and var not in config_dict['dict_col'] and var not in config_dict['vector_col']
                    and var not in config_dict['non_dec_col'] and var not in config_dict['sentence_col']
                    and data[var].nunique() < config_dict['discrete_thresh_hold']]
        # 연속형 변수: 숫자형이면서 이산형이 아닌 경우 및 날자컬럼이 아닌 경우
        continuous = [var for var in data.columns if
                      data[var].dtype != 'O' and var != Y_COL and var not in config_dict['date_col']
                      and var not in config_dict['dict_col'] and var not in config_dict['vector_col']
                      and var not in config_dict['non_dec_col'] and var not in config_dict['sentence_col']
                      and var not in discrete]

    # categorical
    # 객체형(문자열) 데이터이면서 타겟변수가 아닌 경우
    categorical = [var for var in data.columns if
                   data[var].dtype == 'O' and var != Y_COL and var not in config_dict['date_col']
                   and var not in config_dict['dict_col'] and var not in config_dict['vector_col']
                   and var not in config_dict['non_dec_col'] and var not in config_dict['sentence_col']]
    
    # 전처리 데이터 타입 확인용
    print(f'There are {date_cols_len} date_time variables')
    print(f'There are {dict_cols_len} dict variables')
    print(f'There are {vector_cols_len} vector variables')
    print(f'There are {non_dec_cols_len} non-decimal variables')
    print(f'There are {sentence_cols_len} sentence variables')
    print(f'There are {len(discrete)} discrete variables')
    print(f'There are {len(continuous)} continuous variables')
    print(f'There are {len(categorical)} categorical variables')
    return discrete, continuous, categorical


def separate_mixed(df):
    df = df.copy()    
    s = config_dict['mixed_str'][0]
    e = config_dict['mixed_str'][1]
    mixed_col = config_dict['mixed'][0]
    insert_at = df.columns.get_loc(mixed_col)
    num_col = df[mixed_col].str.extract(r'(\d+)').astype('float')
    num_col.columns = [mixed_col + 'num']
    cat_col = df[mixed_col].str[s:e]
    cat_col.name = mixed_col + 'cat'
    df.drop([mixed_col], axis=1, inplace=True)
    left = df.iloc[:, :insert_at]
    right = df.iloc[:, insert_at:]
    df = pd.concat([left, num_col, cat_col.to_frame(), right], axis=1)
    cols = position_Y_COL(list(df.columns))
    return df[cols]

# 소수형을 정수형으로
def truncate_to_integer(series):
    # 모든 값이 1보다 클 때까지 10을 곱함
    while (series < 1).any():
        series *= 10
    
    # 소수점 이하 잘라내고 정수로 변환
    truncated_series = series.astype(int)
    
    # 원본 값과 변환된 값의 관계 저장(XAI 필요하면 사용)
    #value_map = pd.Series(truncated_series.values, index=series.values)
    return truncated_series

# 정수형의 1의 자리를 버림
def truncate_to_ten(series):
    series /= 10
    # 소수점 이하 잘라내고 정수로 변환
    truncated_series = series.astype(int)
    truncated_series *= 10
    # 원본 값과 변환된 값의 관계 저장(XAI 필요하면 사용)
    #value_map = pd.Series(truncated_series.values, index=series.values)
    return truncated_series

def discretiser(df, numeric):
    df = df.copy()
    method = config_dict['discretiser_type'][0]
    cols = config_dict['discretiser']
    for col in cols:  # 각 열에 대해 반복
        if method == 'equalwidth':
            trans = EqualWidthDiscretiser()
            X = df[[col]]
            trans.fit(X)
            df[col] = trans.transform(X)[col]
        elif method == 'equalfrequency':
            trans = EqualFrequencyDiscretiser()
            X = df[[col]]
            trans.fit(X)
            df[col] = trans.transform(X)[col]
        elif method == 'equalfixed':
            
            # 실수형이면 truncate_to_integer 함수 호출
            if np.issubdtype(df[col].dtype, np.floating):  # 실수형 확인
                truncated_data = truncate_to_integer(df[col])
                df[col] = truncated_data  # 변환된 정수형 데이터로 대체
            # 정수형이면 truncate_to_ten 함수 호출
            else:
                truncated_data = truncate_to_ten(df[col])
                df[col] = truncated_data  # 변환된 정수형 데이터로 대체
        else:
            print(f'Method Not Available for column {col}')

        # 실수형이면 truncate_to_integer 함수 호출
        if np.issubdtype(df[col].dtype, np.floating):  # 실수형 확인
            truncated_data = truncate_to_integer(df[col])
            df[col] = truncated_data  # 변환된 정수형 데이터로 대체

    return df

#안씀 - make_imputer_pipe 로 대체
def ohe(df):
    df = df.copy()   
    cols = config_dict['ohe']
    for col in cols:
        trans = OneHotEncoder()
        X = df[[col]]
        trans.fit(X)
        df[col] = trans.transform(X)[col]
    return df

#이상치 탐색을 위한 평균값 구하기
def find_boundaries(df, variable, distance):
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)
    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
    upper_boundary = df[variable].quantile(0.75) + (IQR * distance)
    return upper_boundary, lower_boundary

#이상치 제거 함수
def outlier(df):
    df = df.copy()
    cols = config_dict['outlier']
    for c in cols:
        upper_limit, lower_limit = find_boundaries(df, c, config_dict['iqr'])
        outliers_ = np.where(df[c] > upper_limit, True,
                    np.where(df[c] < lower_limit, True, False))
        df = df.loc[~(outliers_)]
    return df    

#json 데이터 처리
def extract_json_data(df):
    """
    주어진 JSON 문자열 데이터를 파싱하여 딕셔너리 형태로 변환하는 함수

    Parameters:
    json_column_data (list): JSON 문자열이 포함된 리스트

    Returns:
    DataFrame: JSON 데이터가 포함된 새로운 DataFrame
    """
    cols = config_dict['dict_col']
    df = df.copy()
    for col in cols:
        json_column_data = df[col].tolist()
        # JSON 데이터를 저장할 리스트
        json_records = []

        for json_str in json_column_data:
            try:
                # JSON 문자열을 파싱
                json_data = json.loads(json_str)
                json_records.append(json_data)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"JSON 파싱 오류: {str(e)}")
                json_records.append({})  # 오류 발생 시 빈 딕셔너리 추가
        
        # JSON 데이터를 DataFrame으로 변환
        json_df = pd.DataFrame(json_records)

        # 새로운 컬럼명 생성: 기존 컬럼명 + "_" + JSON 키
        new_column_names = {key: f"{col}_{key}" for key in json_df.columns}
        
        # 새로운 컬럼명으로 DataFrame의 컬럼명 변경
        json_df.rename(columns=new_column_names, inplace=True)
        insert_at = df.columns.get_loc(col)
        df.drop(columns=[col], inplace=True)
        left = df.iloc[:, :insert_at]
        right = df.iloc[:, insert_at:]
        df = pd.concat([left, json_df, right], axis=1)
    return df
# 새로운 클래스: 벡터형 → 소수형 (PCA)
class VectorPCAProcessor:
    def __init__(self, variables, n_components=3):
        self.variables = variables
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
    
    def fit(self, X, y=None):
        for col in self.variables:
            # 벡터형 데이터를 numpy 배열로 변환
            vectors = [eval(vec) if isinstance(vec, str) else vec for vec in X[col]]
            vectors = np.array(vectors)
            self.pca.fit(vectors)
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.variables:
            # 벡터형 데이터를 numpy 배열로 변환
            vectors = [eval(vec) if isinstance(vec, str) else vec for vec in X[col]]
            vectors = np.array(vectors)
            # PCA 적용
            transformed = self.pca.transform(vectors)
            # 새로운 컬럼 추가
            for i in range(self.n_components):
                X[f'{col}_pca_{i}'] = transformed[:, i]
            # 원본 컬럼 삭제
            X.drop(columns=[col], inplace=True)
        return X

# 새로운 함수: 진법형 → 정수형
def convert_non_decimal(df):
    df = df.copy()
    cols = config_dict.get('non_dec_col', [])
    if not cols or pd.isna(cols):
        return df
    for col in cols:
        def parse_non_decimal(val):
            try:
                if isinstance(val, str):
                    val = val.lower().strip()
                    if val.startswith('0b'):
                        return int(val, 2)  # 2진수
                    elif val.startswith('0x'):
                        return int(val, 16)  # 16진수
                    else:
                        return int(val)  # 10진수 가정
                return val
            except (ValueError, TypeError):
                return np.nan
        insert_at = df.columns.get_loc(col)
        new_col = df[col].apply(parse_non_decimal)
        new_col.name = f'dec_{col}'
        df.drop(columns=[col], inplace=True)
        left = df.iloc[:, :insert_at]
        right = df.iloc[:, insert_at:]
        df = pd.concat([left, new_col.to_frame(), right], axis=1)
    cols = position_Y_COL(list(df.columns))
    return df[cols]

# 새로운 함수: 문장형 → 벡터형
def sentence_to_vector(df):
    df = df.copy()
    cols = config_dict.get('sentence_col', [])
    if not cols or pd.isna(cols):
        return df
    model_name = config_dict.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
    model = SentenceTransformer(model_name)
    for col in cols:
        insert_at = df.columns.get_loc(col)
        sentences = df[col].fillna('').tolist()
        embeddings = model.encode(sentences, show_progress_bar=False)
        # 벡터를 새로운 컬럼으로 추가
        new_cols = {f"{col}_vec_{i}": embeddings[:, i] for i in range(embeddings.shape[1])}
        new_df = pd.DataFrame(new_cols, index=df.index)  # 원래 인덱스 유지

        # 기존 컬럼 제거
        df.drop(columns=[col], inplace=True)

        # 기존 위치에 삽입
        left = df.iloc[:, :insert_at]
        right = df.iloc[:, insert_at:]
        df = pd.concat([left, new_df, right], axis=1)
    cols = position_Y_COL(list(df.columns))
    return df[cols]

# # 새로운 함수: 문장형 → 벡터형
# def sentence_to_vector(df):
#     df = df.copy()
#     cols = config_dict.get('sentence_col', [])
#     if not cols or pd.isna(cols):
#         return df
#     model_name = config_dict.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
#     model = SentenceTransformer(model_name)
#     for col in cols:
#         sentences = df[col].fillna('').tolist()
#         vector_records = []
#         embeddings = model.encode(sentences, show_progress_bar=False)
#         # 벡터를 새로운 컬럼으로 추가
#         for i in range(embeddings.shape[1]):
#             vector_records.append(embeddings[:, i])
#         vector_df = pd.DataFrame(vector_records)
#         new_column_names = {key: f"{col}_{key}" for key in vector_df.columns}
#         df.drop(columns=[col], inplace=True)
#     cols = position_Y_COL(list(df.columns))
#     return df[cols]


#데이터 전처리 함수
def organize_data(df, y_null_exist):
    df = df.copy()
    cols = list(df.columns)
    cols.remove(Y_COL)
    null_threshhold_cols = []
    discrete, continuous, categorical = discrete_cont(df)
    # - discrete: 이산형 변수
    # - continuous: 연속형 변수
    # - categorical: 범주형 변수

    #널 비율이 임계치를 넘은 컬럼 명세 작성
    for col in cols:
        null_mean = df[col].isnull().mean() # 각 컬럼의 null 비율 계산
        if null_mean >= config_dict['null_threshhold']:
            null_threshhold_cols.append(col)

    #임계치를 넘은 대상을 빼고 다시 df 만듬
    cols_stayed = [c for c in cols if c not in null_threshhold_cols]
    df = df[cols_stayed+[Y_COL]].copy()

    if y_null_exist:
        df = df[df[Y_COL] != df[Y_COL].max()].copy()

    return df, discrete, continuous, categorical

def make_train_test(df):
    df = df.copy()
    X = df.drop(columns=Y_COL)
    y = df[Y_COL]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config_dict['test_size'], random_state=0, stratify=y)
    return X_train, X_test, y_train, y_test

#각 형태별로 파이프라인을 추가하는 형태로 변경
def make_imputer_pipe_old(continuous, discrete, categorical, null_impute_type):
    # 연속형 변수와 이산형 변수를 합쳐서 수치형 변수로 처리
    numberImputer = continuous + discrete

    categoricalImputer = categorical.copy()
    # One-Hot Encoding 대상 변수 제외
    categoricalImputer = [item for item in categoricalImputer if (item not in config_dict['ohe']) ]
    oheImputer = config_dict['ohe']
    
    result={}
    
    # 수치형 변수와 범주형 변수가 모두 있는 경우
    if (len(numberImputer) > 0) & (len(categoricalImputer) > 0):
        pipe = Pipeline([
            # 수치형 변수 결측치 대체
            ("imputer",
            mm.MeanMedianImputer2(
                imputation_method=null_impute_type, variables=numberImputer),),
            # 범주형 변수 결측치 대체
            ('imputer_cat',
            mdi.CategoricalImputer(variables=categorical)),
            # One-Hot Encoding 적용
            ('categorical_encoder',
            ce.OneHotEncoder(variables=oheImputer)),
            # 라벨링 인코딩 적용
            ('categorical_encoder2',
            ce.OrdinalEncoder(encoding_method='ordered',
                variables=categoricalImputer))
        ])
    else:
        # 수치형 변수만 있고 범주형 변수가 없는 경우
        if (len(numberImputer) > 0) & (len(categoricalImputer) == 0):
            pipe = Pipeline([
                # 수치형 변수 결측치만 대체
                ("imputer",
                mm.MeanMedianImputer2(
                    imputation_method=null_impute_type, variables=numberImputer),)
            ])
        else:
            # 범주형 변수만 있고 수치형 변수가 없는 경우
            if (len(numberImputer) == 0) & (len(categoricalImputer) > 0):
                pipe = Pipeline([
                    # 범주형 변수 결측치 대체
                    ('imputer_cat',
                    mdi.CategoricalImputer(variables=categorical)),
                    ('categorical_encoder',
                    ce.OneHotEncoder(variables=oheImputer)),
                    ('categorical_encoder2',
                    ce.OrdinalEncoder(encoding_method='ordered',
                        variables=categoricalImputer))
                ])
            else:
                pipe = []
    return pipe

def make_imputer_pipe(continuous, discrete, categorical, null_impute_type):
    # 연속형 변수와 이산형 변수를 합쳐서 수치형 변수로 처리
    numberImputer = continuous + discrete

    categoricalImputer = categorical.copy()
    # One-Hot Encoding 대상 변수 제외
    categoricalImputer = [item for item in categoricalImputer if (item not in config_dict['ohe']) ]
    oheImputer = config_dict['ohe']
    datecolImputer =  config_dict['date_col'] if config_dict['date_col'] and not pd.isna(config_dict['date_col'][0]) else []
    vectorImputer = config_dict.get('vector_col', []) if config_dict.get('vector_col', []) and not pd.isna(config_dict.get('vector_col', [])[0]) else []
    # result={}
    
    steps = []
    # 수치형 변수 처리 파이프라인(결측치를 null_impute_type값[mean,median,max,min]에 따라 채움)
    if numberImputer  and len(numberImputer) > 0:
        steps.append(("numeric_imputer", mm.MeanMedianImputer2(imputation_method=null_impute_type, variables=numberImputer)))
        # print('processing numeric')

    # 범주형 변수 처리 파이프라인(결측치를 최빈값으로 채움)
    if categorical  and len(categorical) > 0:
        steps.append(('categorical_imputer', mdi.CategoricalImputer(variables=categorical)))
        # print('processing categorical')

    # 원핫인코딩 처리(데이터 종류만큼 컬럼을 만들어 1,0으로 표현)
    if oheImputer and len(oheImputer) > 0:
        steps.append(('onehot_encoder', ce.OneHotEncoder(variables=oheImputer)))
        # print('processing ohe')

    # 라벨 인코딩 처리(데이터 종류별 고유수치로 변경, male->1, female->2)
    if categoricalImputer and len(categoricalImputer) > 0:
        steps.append(('label_encoder', CustomOrdinalEncoder(encoding_method='ordered', variables=categoricalImputer)))
        # print('processing label encoding')


    # 시계열 데이터 처리(날짜형에서 연월일 추출, 시간형에서 타임델타 추출)
    if datecolImputer and len(datecolImputer) > 0:
        steps.append(('temporal_feature_engineering', tf.DateFeatureTransformer2(variables=datecolImputer, features=['year', 'month', 'day', 'time_seconds'], drop_original=True)))
        # print('processing datecol imputing')
    
    # 벡터 데이터 처리
    if vectorImputer and len(vectorImputer) > 0:
        steps.append(('vector_pca', VectorPCAProcessor(variables=vectorImputer, n_components=config_dict.get('pca_components', 3))))
        # print('processing vector imputing')
    
    # 파이프라인 생성
    return Pipeline(steps) if steps else []

def restore_column_order(df_transformed, original_order):
    df_transformed = df_transformed.copy()

    # 원래 있던 컬럼 순서를 최대한 앞에 배치
    ordered_cols = [col for col in original_order if col in df_transformed.columns]

    # 나머지 새로 생성된 컬럼은 뒤쪽에 정렬
    remaining_cols = [col for col in df_transformed.columns if col not in ordered_cols]

    return df_transformed[ordered_cols + remaining_cols]


def do_imputation(df, pipe, original_order):
    train=False
    if(train):
        xtrain, xtest, y_train, y_test = make_train_test(df)
        
        # pipe.fit(X_train, y_train)
        # 파이프라인을 훈련 데이터에 맞춤
        pipe.fit(xtrain, y_train)
        X_train = pipe.transform(xtrain)
        X_test = pipe.transform(xtest)

        # 컬럼 복원
        X_train = restore_column_order(X_train, original_order)
        X_test = restore_column_order(X_test, original_order)

        # 훈련 세트에 타겟 변수와 'split' 열 추가
        X_train[Y_COL] = y_train        
        X_train['split'] = 'train'
        # 테스트 세트에 타겟 변수와 'split' 열 추가
        X_test[Y_COL] = y_test
        X_test['split'] = 'test'        
        return pd.concat([X_train, X_test]).reset_index(drop=True)
    else:
        # 전체 데이터에 대해 파이프라인을 적용
        
        # 타겟 변수 분리
        y_full = df[Y_COL]
        
        # 파이프라인을 전체 데이터에 맞춤
        pipe.fit(df.drop(columns=[Y_COL]),y_full)
        
        # 변환 적용
        X_full = pipe.transform(df.drop(columns=[Y_COL]))

        # 컬럼 순서 복원 시도
        X_full = restore_column_order(X_full, original_order)
        # 변환된 데이터프레임에 타겟 변수 추가
        X_full[Y_COL] = y_full
        X_full['split'] = 'full'
        
        return X_full.reset_index(drop=True)

def scaling(df):    
    df = df.copy()
    if config_dict['scale'] is np.nan:
        config_dict['scale'] = ['minmax']   # default with minmax scaling
    if config_dict['scale'][0] =='minmax':
        scaler = MinMaxScaler()
        scaler.fit(df)
        return scaler.transform(df)
    elif config_dict['scale'][0] =='standard':
        scaler = StandardScaler()
        scaler.fit(df)
        return scaler.transform(df)
    else: 
        scaler = MinMaxScaler()
        scaler.fit(df)
        return scaler.transform(df)        


if __name__ == '__main__':
    # arv 예1: credit 
    # arv 예2: metro 

    try:
        #파라미터
        #folder = sys.argv[1]    # take input with argv parameter
        folder = "loans"    #테스트용

        parent = join_abs_path(os.getcwd(), os.pardir)
        conf_file = f'argumet_{folder}.xlsx'      
        configs = pd.read_excel(join_abs_path(f'{parent}/config', conf_file), header=None).set_index(0)        
        config_cols = configs.index.tolist()
        config_dict = {}
        for c in config_cols:
            config_dict[c] = configs.loc[c].values[0]
            if (type(config_dict[c]) == int) or (type(config_dict[c]) == float):
                pass
            else:
                config_dict[c] = configs.loc[c].values[0].split(',')
        ori_file_name = config_dict['file_name'][0].split('.')[0]
        
        if config_dict['mixed_str'] is np.nan or len(config_dict['mixed_str']) < 1:
            pass
        else:
            config_dict['mixed_str'] = [eval(i) for i in config_dict['mixed_str']]  #배열의 각 값을 정수형으로 변환

        if config_dict['y_col'] is np.nan or len(config_dict['y_col']) != 1:
            print('No Y column exists')
            raise Exception

        if config_dict['discrete_thresh_hold'] is np.nan or config_dict['discrete_thresh_hold'] < 0:
            print('discrete_thresh_hold set to default 10')
            config_dict['discrete_thresh_hold'] = 10

        Y_COL = config_dict['y_col'][0]
        original_file = join_abs_path(f'{parent}/data/{folder}', config_dict['file_name'][0])
        df_initial = read_data(original_file)
        original_order = df_initial.drop(columns=[Y_COL]).columns.tolist()  # 원본 컬럼 순서 저장

        # 1. Label column Encoding
        df_labeld, y_null_exist = y_label_enc(df_initial)

        # 1.1json 처리
        df_jsoned = extract_json_data(df_labeld)

        # 1.2 진법형 처리
        df_non_dec = convert_non_decimal(df_jsoned)

        # 1.3 문장형 처리
        df_sentenced = sentence_to_vector(df_non_dec)

        # 2. 데이터 정리 및 변수 분류
        df_organized, discrete, continuous, categorical = organize_data(df_sentenced, y_null_exist)

        # 3. Mixed 칼럼을 숫자형/문자형으로 분리(분리 후 df_organized, discrete, continuous, categorical 재분류)
        if config_dict['mixed'] is not np.nan:
            df = separate_mixed(df_organized)
            discrete, continuous, categorical = discrete_cont(df)
        else:
            df = df_organized.copy()

        # null_impute_types 정의
        null_impute_types = config_dict['null_imp']

        if null_impute_types is not np.nan:
            for null_impute_type in null_impute_types:
        # 4. pipeline 정의
                print("🚧 Before calling make_imputer_pipe")
                pipe = make_imputer_pipe(discrete, continuous, categorical, null_impute_type)
                print("✅ Created pipeline")
                if pipe == []:
                    print('no pipe applied')
                else:
        # 5. discretization(연속형 변수를 범주형으로)
                    if config_dict['discretiser'] is not np.nan:
                        df_piped = discretiser(df, discrete+continuous)
        # 6. imputation thru pipeline
                    print("🚧 Before calling do_imputation")
                    df_piped = do_imputation(df, pipe,original_order)
                    print("✅ Finished imputation")
                    dest_path = os.path.join(parent, os.path.join('data_preprocessed', f'{folder}'))
                    dest_path = os.path.join(parent, os.path.join(dest_path, f'{dest_path}/imputed'))
                    Path(dest_path).mkdir(parents=True, exist_ok=True)
                    dest_path = os.path.join(parent, os.path.join(dest_path, f'imputed_{ori_file_name}_{null_impute_type}.csv'))
                    df_piped.to_csv(dest_path, index=False)

        # 8. discretization(연속형 변수를 범주형으로)
                    if config_dict['discretiser'] is not np.nan:
                        df_piped = discretiser(df_piped, discrete+continuous)
        
        # 9. Outlier 처리
                    if config_dict['outlier'] is not np.nan:    
                        df_piped = outlier(df_piped)
                        df_piped = df_piped.reset_index(drop=True)
        # 9.1 데이터 정제 저장
                    dest_path = os.path.join(parent, os.path.join('data_preprocessed', f'{folder}'))
                    dest_path = os.path.join(parent, os.path.join(dest_path, 'trans'))
                    Path(dest_path).mkdir(parents=True, exist_ok=True)
                    dest_path = os.path.join(parent, os.path.join(dest_path, f'trans_{ori_file_name}_{null_impute_type}.csv'))
                    df_piped.to_csv(dest_path, index=False)


        # 10. 스케일링 작업 및 저장/ Train과 Test 를 따로 스케일링
        # 10.1 X_train 스케일링
                    con = df_piped['split'] == 'train'
                    if not df_piped[con].empty:
                        X_train_scaled = scaling(df_piped[con].drop(columns=[Y_COL,'split']))
                        X_train_scaled = pd.DataFrame(X_train_scaled)
                        X_train_scaled[Y_COL] = df_piped[con][Y_COL]
                        X_train_scaled['split'] = df_piped[con]['split']
                        X_train_scaled.columns = df_piped.columns
                    else:
                        X_train_scaled = []
        # 10.2 X_test 스케일링
                    con = df_piped['split'] == 'test'
                    X_test_scaled = []
                    if not df_piped[con].empty:
                        X_test_scaled = scaling(df_piped[con].drop(columns=[Y_COL,'split']))
                        X_test_scaled = pd.DataFrame(X_test_scaled)
                        tmp = df_piped.copy().reset_index()
                        X_test_scaled['index'] = tmp[con]['index'].values
                        X_test_scaled = X_test_scaled.set_index('index')
                        X_test_scaled[Y_COL] = df_piped[con][Y_COL]
                        X_test_scaled['split'] = df_piped[con]['split']
                        X_test_scaled.columns = df_piped.columns
                        X_test_scaled.index.name = None
                        del tmp
                    else:
                        X_test_scaled = []        
        # 10.3 data frame merge                                        
                    if (len(X_train_scaled) == 0 and len(X_test_scaled) == 0 ):
                        df_scaled = scaling(df_piped.drop(columns=[Y_COL,'split']))
                        df_scaled = pd.DataFrame(df_scaled)
                        df_scaled[Y_COL] = df_piped[Y_COL]
                        df_scaled['split'] = df_piped['split']
                        df_scaled.columns = df_piped.columns
                    else :
                        df_scaled = pd.concat([X_train_scaled, X_test_scaled])
        # 10.4 scaling 저장                        
                    dest_path = os.path.join(parent, os.path.join('data_preprocessed', f'{folder}'))
                    dest_path = os.path.join(parent, os.path.join(dest_path, 'scaled'))
                    Path(dest_path).mkdir(parents=True, exist_ok=True)
                    dest_path = os.path.join(parent, os.path.join(dest_path, f'scaled_{ori_file_name}_{null_impute_type}.csv'))
                    df_scaled.to_csv(dest_path, index=False)
        print('Completed.')
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print('비정상종료', e)
        traceback.print_exc()
        print(exc_type, exc_tb.tb_lineno)