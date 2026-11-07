
# Reproduction Guide

## Requirements

Python 3.11 또는 3.12 환경이 필요합니다.

저장소의 전체 실험 결과는 Windows / Python 3.11.3 환경에서 생성했으며, GitHub Actions는 Linux / Python 3.12 환경에서 자동화 테스트를 수행합니다.

프로젝트 루트에서 의존성을 설치합니다.

```bash
python -m pip install -r requirements.txt
```

별도의 가상환경을 사용하는 것을 권장합니다.

## Run Tests

단위 테스트와 합성 데이터 스모크 테스트는 실제 Telco 데이터셋이나 네트워크 연결 없이 실행할 수 있습니다.

```bash
python -m unittest discover -s tests -v
python -m scripts.smoke
```

단위 테스트는 데이터 정제, 데이터 분할, Fold별 전처리 및 특성 선택, 모델 입력 정합성, 평가 지표 등을 검증합니다.

스모크 테스트는 합성 데이터로 학습·평가 파이프라인을 실행하며, 전체 81개 하이퍼파라미터 조합을 탐색하지는 않습니다.

## Download the Dataset

IBM이 공개한 Telco Customer Churn CSV를 다운로드합니다.

```bash
python -m scripts.download_data --output data/Telco-Customer-Churn.csv
```

다운로드 스크립트는 데이터 구조를 검증하고 SHA-256 및 출처 메타데이터를 기록합니다. 이미 존재하는 파일은 덮어쓰지 않습니다.

기존 실험과 동일한 데이터인지 확인하려면 다음 명령을 실행하세요.

```powershell
python -c "import hashlib; from pathlib import Path; p=Path('data/Telco-Customer-Churn.csv'); actual=hashlib.sha256(p.read_bytes()).hexdigest(); expected='16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91'; print('SHA-256:', actual); assert actual == expected, 'Dataset checksum mismatch'"
```

체크섬이 일치하지 않으면 기존 실험에 사용한 데이터와 바이트가 다르다는 뜻입니다. 이 경우 데이터 출처와 버전을 확인한 뒤 재실험 여부를 판단해야 합니다.

이미 보유한 호환 CSV도 `--data-path` 옵션으로 지정할 수 있지만, 기존 결과의 재현을 주장하려면 데이터 동일성을 별도로 확인해야 합니다.

## Run the Experiment

전체 실험을 실행합니다.

```bash
python -m scripts.run_experiment --data-path data/Telco-Customer-Churn.csv --seed 42 --output-dir results-rerun --n-jobs 1
```

이 명령은 다음을 수행합니다.

- Stratified 70:30 훈련·테스트 분할
- 훈련 데이터 내부의 Stratified 5-Fold 교차검증
- Fold별 전처리 및 Logistic Regression 기반 RFE
- Decision Tree 72개 하이퍼파라미터 조합 탐색
- LinearSVC 9개 하이퍼파라미터 조합 탐색
- 기본 분류기 및 비교 모델을 포함한 총 7개 모델 평가
- 결과 지표, 선택 특성, 탐색 결과 및 시각화 저장

출력 디렉터리는 비어 있어야 합니다. 기존 `results/`의 보존을 위해 새로운 디렉터리를 지정하세요.

전체 실험 대신 일부 모델만 실행하려면 `--models` 옵션을 사용할 수 있습니다.

지원하는 그룹은 `decision_tree`, `linear_svm`, `rbf_svm`, `baselines`입니다.

## Validate Results

새로 생성한 전체 실험 결과를 검증합니다.

```bash
python -m scripts.validate_artifacts results-rerun --full
```

저장소에 커밋된 기존 결과도 동일한 방법으로 확인할 수 있습니다.

```bash
python -m scripts.validate_artifacts results --full
```

검증 대상은 다음과 같습니다.

- 실험 완료 상태와 모델 수
- Decision Tree 72개 및 LinearSVC 9개 탐색 결과
- 모델별 평가 지표와 선택 특성 수
- 혼동행렬과 테스트 표본 수의 일치 여부
- 저장된 산출물의 SHA-256
- 현재 소스코드와 실험 당시 기록된 소스 SHA-256의 일치 여부

이 검증은 저장된 산출물의 정합성을 확인하는 절차이며, 전체 모델을 다시 학습하지 않습니다.

## Experiment Configuration

**데이터 분할:** Stratified 70% Train / 30% Test, Seed 42

**교차검증:** 훈련 데이터 내부의 Stratified 5-Fold CV

**수치형 전처리:** StandardScaler

**범주형 전처리:** OneHotEncoder

**특성 선택:** Logistic Regression 기반 RFE로 20개 특성을 선택하며, 학습은 각 CV 훈련 Fold 내부에서 수행

**모델 선택:** 평균 CV Average Precision(AP)이 가장 높은 설정 선택. 점수가 정확히 동일하면 ParameterGrid의 첫 번째 설정 사용

**최종 평가:** 모델 설정을 확정한 뒤 분리된 테스트셋에서 수행

**비교 모델:** DummyClassifier와 Logistic Regression은 RFE 없이 전처리된 전체 특성을 사용

LinearSVC는 `dual="auto"`로 설정합니다. 별도의 결정 임계값 튜닝은 수행하지 않습니다.

학습 과정의 경고는 `warnings.json`에 기록합니다. `--n-jobs 1`은 단일 프로세스에서 경고를 수집할 때 사용하는 권장 설정입니다.

실행 환경, 라이브러리 버전, 소스코드 및 산출물 SHA-256은 `experiment_manifest.json`에 기록됩니다. 실행 환경이나 라이브러리 버전이 달라지면 수치 결과가 일부 달라질 수 있습니다.

## Continuous Integration

GitHub Actions는 Python 3.12 환경에서 다음을 실행합니다.

- 단위 테스트
- 합성 데이터 스모크 테스트
- 커밋된 실험 산출물 검증
- Git 변경사항의 공백 오류 검사

전체 81개 하이퍼파라미터 탐색은 CI에서 반복 실행하지 않습니다.

현재 검증기는 소스코드 SHA-256을 검사하므로, 학습·평가 코드를 변경한 경우에는 기존 실험 결과와의 정합성을 다시 확인해야 합니다.
