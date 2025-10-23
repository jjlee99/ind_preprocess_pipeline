# 데이터 전처리 파이프라인 (preprocess_V1.6_stable.py)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/라이선스-MIT-green.svg)
![Status](https://img.shields.io/badge/상태-안정화-brightgreen.svg)

## 📖 개요

`preprocess_V1.6_stable.py`는 원시 데이터를 머신러닝 작업에 적합하게 준비하기 위해 설계된 강력하고 모듈화된 데이터 전처리 파이프라인입니다. 숫자형, 범주형, 날짜/시간, JSON, 벡터, 텍스트 데이터를 포함한 다양한 데이터 유형을 처리하며, 유연성, 확장성, 사용 편의성에 중점을 둡니다. 이 파이프라인은 설정 파일을 통해 동적으로 작동하여 특정 데이터셋에 맞춘 전처리가 가능합니다.


---

## ✨ 주요 기능

- **포괄적인 데이터 처리**:
  - 숫자형(이산형/연속형), 범주형, 날짜/시간, JSON, 벡터, 텍스트 데이터 지원.
  - JSON 파싱, 문장 임베딩 등 특수 데이터 유형을 위한 사용자 정의 변환기 제공.
- **모듈화된 파이프라인**:
  - `scikit-learn`의 `Pipeline`을 사용하여 전처리 단계를 원활히 통합.
  - 결측치 대체, 인코딩, 이산화, 이상치 제거, 스케일링 포함.
- **설정 파일 기반**:
  - Excel 설정 파일(`argumet_{folder}.xlsx`)을 통해 동적 컬럼 선택 및 전처리 설정 제공.
- **출력 관리**:
  - 전처리 단계별 데이터(`imputed`, `transformed`, `scaled`)를 저장하여 추적 가능.
- **에러 처리**:
  - 안정적인 실행을 위한 강력한 예외 처리 및 로그 출력.

---

## 🛠️ 설치

### 전제 조건
- Python 3.8 이상
- 필요한 라이브러리:
  ```bash
  pip install pandas numpy scikit-learn feature_engine sentence-transformers
  ```

### 설정
1. 리포지토리를 클론:
   ```bash
   git clone https://github.com/your-repo/data-preprocessing-pipeline.git
   ```
2. 의존성 설치:
   ```bash
   pip install -r requirements.txt
   ```
3. `config/` 디렉토리에 설정 파일(`argumet_{folder}.xlsx`)을 준비.

---

## 🚀 사용 방법

### 1. 설정 파일 준비
Excel 파일(예: `argumet_loans.xlsx`)을 다음 구조로 작성:

| 설정 키                 | 예시 값                           | 설명                                     |
|-------------------------|-----------------------------------|------------------------------------------|
| `file_name`             | `loans.csv`                      | 입력 CSV 파일 이름                      |
| `y_col`                 | `target`                         | 타겟 컬럼 이름                          |
| `keep_col`              | `age,income,target`              | 유지할 컬럼                             |
| `date_col`              | `signup_date`                    | 날짜/시간 컬럼                          |
| `dict_col`              | `user_info`                      | JSON 컬럼                               |
| `sentence_col`          | `description`                    | 임베딩용 텍스트 컬럼                    |
| `scale`                 | `minmax`                         | 스케일링 방법 (`minmax` 또는 `standard`) |
| `null_imp`              | `mean`                           | 숫자형 데이터 결측치 대체 방법           |
| `discrete_thresh_hold`  | `10`                             | 이산형 변수 임계값                      |
| `mixed`                 | `mixed_col`                      | 혼합형 컬럼                             |
| `mixed_str`             | `0,3`                            | 혼합형 컬럼 분리용 시작/끝 인덱스       |

### 2. 파이프라인 실행
데이터셋 폴더 이름을 인수로 스크립트를 실행:
```bash
python preprocess_V1.6_stable.py loans
```

### 3. 출력
전처리된 데이터는 다음 디렉토리에 저장:
- `data_preprocessed/loans/imputed/`: 결측치 대체 데이터
- `data_preprocessed/loans/trans/`: 변환 데이터 (이산화/이상치 제거 후)
- `data_preprocessed/loans/scaled/`: 스케일링된 데이터

예시 출력 파일:
- `imputed_loans_mean.csv`
- `trans_loans_mean.csv`
- `scaled_loans_mean.csv`

---

## 📋 파이프라인 워크플로우

파이프라인은 다음 단계를 통해 데이터를 처리:

1. **데이터 로드**:
   - 설정 파일의 `keep_col`에 지정된 컬럼을 로드.
   - 타겟 컬럼(`y_col`)을 데이터프레임 끝으로 재배치.

2. **타겟 인코딩**:
   - `LabelEncoder`를 사용해 타겟 변수를 숫자형으로 인코딩.
   - 타겟 변수의 결측치 여부 추적.

3. **데이터 정리**:
   - 결측치 비율이 높은 컬럼(`null_threshhold`) 제거.
   - 컬럼을 이산형, 연속형, 범주형으로 분류.

4. **혼합형 컬럼 처리**:
   - 혼합형 컬럼(예: `ABC123XYZ`)을 숫자형과 범주형으로 분리.

5. **전처리 파이프라인**:
   - **결측치 대체**: 숫자형(평균/중앙값), 범주형(최빈값) 결측치 처리.
   - **인코딩**: 범주형 변수에 원핫 또는 순서형 인코딩 적용.
   - **날짜/시간 처리**: 날짜 컬럼에서 연, 월, 일, 시간 등의 피처 추출.
   - **JSON 파싱**: JSON 문자열을 새로운 컬럼으로 확장.
   - **벡터 처리**: PCA를 통해 벡터 데이터 차원 축소.
   - **기타 진수 데이터 변환**: 2진수/16진수 문자열을 10진수로 변환.
   - **텍스트 임베딩**: 텍스트를 `sentence-transformers`로 벡터 임베딩 변환.

6. **이산화**:
   - 연속형 변수를 등폭(equal-width), 등빈도(equal-frequency), 또는 고정값 자르기 방식으로 이산화.

7. **이상치 제거**:
   - 지정된 컬럼에 대해 IQR(사분위수 범위)을 기준으로 이상치 제거.

8. **스케일링**:
   - 설정에 따라 `MinMaxScaler` 또는 `StandardScaler` 적용.

9. **출력 저장**:
   - 각 단계별 데이터를 저장하여 추적 및 분석 가능.

---

## 🧩 사용자 정의 변환기

산업데이터 전처리 파이프라인으로써, 본 파이프라인은 실제로 발생할 수 있는 여러 데이터 유형을 처리하기 위한  `scikit-learn` 기반 커스터마이징 전처리기를 포함하고 있습니다

| 변환기                          | 기능                                       | 입력 → 출력 예시                          |
|--------------------------------|--------------------------------------------|-------------------------------------------|
| `CustomOrdinalEncoder`         | 범주형 데이터를 순서형 인코딩              | `["male", "female"]` → `[0, 1]`           |
| `JSONExtractorTransformer`     | JSON 문자열을 컬럼으로 파싱                 | `{"age": 30}` → `col_age: 30`             |
| `NonDecimalConverterTransformer`| 비10진수 문자열을 10진수로 변환            | `"0x1A"` → `26`                           |
| `SentenceEmbeddingTransformer`  | 텍스트를 벡터 임베딩으로 변환              | `"I love coding"` → `[0.1, 0.2, ...]`     |
| `VectorPCAProcessor`           | 벡터 데이터를 PCA로 차원 축소              | `[1.2, 3.4, 5.6]` → `[0.5, 0.3]`         |

---

## 📊 설정(configuration) 예시

`loans.csv` 데이터셋(컬럼: `age`, `income`, `signup_date`, `user_info`(JSON), `description`(텍스트), `target`)의 경우:

```excel
| 0                    | 1                          |
|----------------------|----------------------------|
| file_name            | loans.csv                  |
| y_col                | target                     |
| keep_col             | age,income,signup_date,user_info,description,target |
| date_col             | signup_date                |
| dict_col             | user_info                  |
| sentence_col         | description                |
| scale                | minmax                     |
| null_imp             | mean                       |
| discrete_thresh_hold | 10                         |
```

**명령어**:
```bash
python preprocess_V1.6_stable.py loans
```

**출력**:
- `data_preprocessed/loans/` 디렉토리에 결측치 대체, 변환, 스케일링된 CSV 파일 저장.

---

## ⚙️ 설정 세부 정보

| 키                      | 설명                                       | 예시 값                           |
|-------------------------|--------------------------------------------|-----------------------------------|
| `file_name`             | 입력 CSV 파일 이름                         | `loans.csv`                      |
| `y_col`                 | 타겟 컬럼                                 | `target`                         |
| `keep_col`              | 유지할 컬럼                               | `age,income,target`              |
| `date_col`              | 날짜/시간 컬럼                            | `signup_date`                    |
| `dict_col`              | JSON 컬럼                                 | `user_info`                      |
| `vector_col`            | PCA용 벡터 컬럼                           | `vector_data`                    |
| `non_dec_col`           | 비10진수(2진수/16진수) 컬럼               | `hex_id`                         |
| `sentence_col`          | 임베딩용 텍스트 컬럼                      | `description`                    |
| `mixed`                 | 분리할 혼합형 컬럼                        | `mixed_col`                      |
| `mixed_str`             | 혼합형 컬럼 분리용 시작/끝 인덱스         | `0,3`                            |
| `discretiser`           | 이산화할 컬럼                             | `age,income`                     |
| `discretiser_type`      | 이산화 방법                               | `equalwidth`                     |
| `ohe`                   | 원핫 인코딩 컬럼                          | `category`                       |
| `outlier`               | 이상치 제거 대상 컬럼                     | `income`                         |
| `null_imp`              | 숫자형 결측치 대체 방법                   | `mean`                           |
| `scale`                 | 스케일링 방법                             | `minmax`                         |
| `embedding_model`       | 문장 임베딩 모델                          | `sentence-transformers/all-MiniLM-L6-v2` |
| `sentence_embedding_dims_to_keep` | 유지할 임베딩 차원 수             | `50`                             |
| `pca_components`        | PCA 주성분 수                             | `3`                              |

---

## 🔍 참고 사항

- **성능**: 중간 규모 데이터셋에 최적화. 대규모 데이터셋의 경우 배치 처리 또는 병렬화 고려.
- **확장성**: 새로운 데이터 유형을 위한 사용자 정의 변환기 추가로 확장 가능.
- **에러 처리**: 포괄적인 로깅과 예외 처리로 안정적인 실행 보장.
- **의존성**: 고급 전처리를 위해 `feature_engine` 및 `sentence-transformers` 필요.

---

## 📬 기여 방법


1. 리포지토리를 포크.
2. 새 브랜치 생성 (`git checkout -b feature/your-feature`).
3. 변경사항 커밋 (`git commit -m 'Add your feature'`).
4. 브랜치 푸시 (`git push origin feature/your-feature`).
5. 풀 리퀘스트 생성.

코드는 PEP 8 표준을 따르고 관련 테스트를 포함해주세요.

