# Reproduction Guide

## Setup

Python 3.11 또는 3.12를 사용합니다. 공개 결과는 Windows·Python 3.11.3에서 생성했고, GitHub Actions는 Linux·Python 3.12에서 오프라인 검사를 실행합니다. 저장소를 복제하고 가상환경을 만듭니다.

~~~bash
git clone https://github.com/tangerineYR/telco-churn-ml.git
cd telco-churn-ml
python -m venv .venv
~~~

Linux·macOS에서는 `source .venv/bin/activate`, Windows PowerShell에서는 `.\.venv\Scripts\Activate.ps1`로 가상환경을 활성화한 뒤 설치합니다.

~~~bash
python -m pip install -r requirements.txt
~~~

이 문서의 Python 명령은 가상환경을 활성화한 터미널에서 저장소 루트를 기준으로 실행합니다. [의존성 버전](requirements.txt)

## Offline Checks

실제 Telco CSV 없이 실행할 수 있습니다.

~~~bash
python -m unittest discover -s tests -v
python -m scripts.smoke
python -m scripts.validate_artifacts results --full
~~~

첫 명령은 단위 테스트 16개를 실행합니다. 두 번째 명령은 합성 데이터에서 트리·LinearSVC 탐색 조합을 각각 하나로 줄여 학습과 평가 흐름을 확인합니다. 세 번째 명령은 공개 실험의 모델 7개, 탐색 기록 81개, 지표·특성·그림·소스 파일을 실험 기록과 대조합니다. 원본 데이터로 지표를 다시 계산하려면 아래 전체 실험을 실행합니다. [CI 워크플로](.github/workflows/ci.yml)

## Dataset

IBM 공개 저장소의 `master`에서 CSV를 받습니다. 다운로드 명령은 기존 CSV 파일을 덮어쓰지 않으며, 21개 열과 정제 조건을 검증한 뒤 옆에 `Telco-Customer-Churn.provenance.json`을 생성합니다.

~~~bash
python -m scripts.download_data --output data/Telco-Customer-Churn.csv
~~~

공개 결과와 같은 버전의 입력이 필요하면 고정된 IBM CSV를 내려받아 같은 위치에 둘 수 있습니다. 이미 확보한 CSV는 실험 명령의 `--data-path`로 지정합니다. CSV 옆에 같은 이름의 `.provenance.json`이 있으면 실험 실행 시 두 파일이 일치하는지 확인합니다. 공개 실험의 원본은 7,043행이며, 정제 후 7,032행으로 평가합니다. [고정된 IBM 원본 CSV](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/1fd6fd70906479be1712e64cfc0ea89f58466629/data/Telco-Customer-Churn.csv) · [출처 기록](historical/dataset_provenance.json)

## Full Experiment

전체 훈련에는 트리 72개와 LinearSVC 9개 조합의 5겹 교차검증 및 모델 7개의 테스트 평가가 포함됩니다. 아래 두 명령으로 결과를 새 폴더에 생성하고 그 결과를 검사합니다.

~~~bash
python -m scripts.run_experiment --data-path data/Telco-Customer-Churn.csv --seed 42 --output-dir results-rerun --n-jobs 1
python -m scripts.validate_artifacts results-rerun --full
~~~

출력 경로에는 새 폴더나 비어 있는 폴더를 지정합니다. `--n-jobs 1`은 공개 실험의 설정이며, 탐색을 한 작업에서 실행해 학습 경고를 `warnings.json`에 수집합니다. 사용한 라이브러리와 실행 환경은 `experiment_manifest.json`에 기록되며, 환경이 다르면 수치가 일부 달라질 수 있습니다.

| 모델 | 탐색한 하이퍼파라미터 | 조합 수 |
| --- | --- | ---: |
| Decision Tree | `criterion`은 `gini`, `entropy`. `max_depth`는 `3`, `5`, `7`, `None`. `min_samples_split`은 `2`, `5`, `10`. `min_samples_leaf`는 `1`, `2`, `4`. | 72 |
| LinearSVC | `C`는 `0.1`, `1`, `10`. `max_iter`는 `1000`, `5000`, `10000`. `dual="auto"`로 고정. | 9 |

시드 `42`의 70:30 층화 분할 뒤, 훈련셋에서 셔플을 적용한 층화 5겹 교차검증을 수행합니다. 검증 평균 AP가 가장 높은 조합을 선택하고, 점수가 같으면 `ParameterGrid` 순서가 앞선 조합을 택합니다. 선택한 설정의 전처리기·RFE·분류기를 훈련셋 전체로 다시 학습한 뒤 테스트셋을 평가합니다. [공개 실험 설정](results/design.json) · [모델과 탐색 코드](telco_churn/models.py)

## Partial Experiment

`--models` 뒤에 실행할 모델 그룹을 한 개 이상 지정할 수 있습니다. 예를 들어 비교 모델 두 개만 학습하고 결과를 검사하려면 아래 명령을 실행합니다.

~~~bash
python -m scripts.run_experiment --data-path data/Telco-Customer-Churn.csv --seed 42 --output-dir work/results-baselines --models baselines --n-jobs 1
python -m scripts.validate_artifacts work/results-baselines
~~~

`--full`은 일곱 모델과 81개 탐색 기록을 요구하는 전체 결과 검사 옵션입니다. 위 비교 모델 실행은 모델 2개·탐색 0개로 검증됩니다.

| 그룹 | 포함되는 모델 |
| --- | --- |
| `decision_tree` | 기본 Decision Tree와 튜닝한 Decision Tree |
| `linear_svm` | 기본 LinearSVC와 튜닝한 LinearSVC |
| `rbf_svm` | RBF SVM |
| `baselines` | 최빈 클래스 모델과 로지스틱 회귀 |

## Output Files

| 파일 | 확인할 내용 |
| --- | --- |
| `design.json` | 배정된 훈련·테스트 행 수, 클래스 분포, 교차검증과 선택 기준 |
| `metrics.json` | 모델별 테스트 지표, 혼동행렬, 사용한 점수 종류, 선택한 설정 |
| `cv_search_results.csv` | 훈련셋 내부의 조합별·폴드별 검증 지표와 순위 |
| `selected_features.json` | 재학습 모델별 특성명과 입력 순서 |
| `experiment_manifest.json` | 패키지 버전, 실행 환경과 설정, 검증에 사용하는 소스·결과 파일 기록 |
| `warnings.json` | 실행 중 수집한 경고 |
| `figures/` | 선택한 모델에 따라 ROC·정밀도–재현율 곡선, 튜닝 트리와 모델별 혼동행렬 |

결과 검증기는 지정한 폴더의 산출물과 현재 소스 파일을 실행 기록과 대조합니다. 학습·평가 코드를 변경한 뒤의 결과를 검증하려면 그 코드에서 실행한 폴더를 지정합니다. [저장 결과 검증 코드](scripts/validate_artifacts.py)