# 프로젝트 개요

## 프로젝트 설명
범용 이미지 유사도 검색 플랫폼 - CLIP 모델 기반 이미지 임베딩 추출 및 유사도 계산 API

**주요 기능**:
- 이미지 업로드 또는 URL을 통한 유사도 검색
- 프로토타입 임베딩과의 코사인 유사도 계산
- 임계값 기반 Unknown 판정 시스템
- 다중 도메인 지원 가능한 확장 가능한 아키텍처

**기술 스택**:
- FastAPI 0.119.0
- PyTorch 2.8.0 + Transformers 4.57.1
- CLIP (openai/clip-vit-base-patch32)
- Pydantic 2.12.2 (validation & settings)
- Uvicorn (ASGI server)

---

# 프로젝트 구조

## 디렉토리 구조
```
image-similarity-server/
├── app/
│   ├── main.py                    # FastAPI 애플리케이션 엔트리포인트
│   ├── api/                       # API 라우팅 레이어
│   │   ├── __init__.py
│   │   └── v1/                    # API 버전 1
│   │       ├── __init__.py
│   │       ├── router.py          # v1 메인 라우터
│   │       └── endpoints/         # 엔드포인트 모듈
│   │           ├── __init__.py
│   │           └── similarity.py  # 유사도 검색 엔드포인트
│   ├── core/                      # 핵심 설정 및 의존성
│   │   ├── __init__.py
│   │   ├── config.py              # Pydantic Settings (환경 설정)
│   │   └── dependencies.py        # 의존성 주입 (DI)
│   ├── models/                    # AI 모델 래퍼
│   │   ├── __init__.py
│   │   ├── embedding_extractor.py # CLIP 임베딩 추출기
│   │   └── similarity_calculator.py # 코사인 유사도 계산기
│   ├── schemas/                   # Pydantic 스키마 (요청/응답)
│   │   ├── __init__.py
│   │   └── similarity.py          # 유사도 API 스키마
│   ├── services/                  # 비즈니스 로직 레이어
│   │   ├── __init__.py
│   │   ├── embedding_service.py   # 임베딩 추출 서비스
│   │   └── similarity_service.py  # 유사도 계산 서비스
│   └── ai_models/                 # AI 모델 데이터 저장소
│       └── pokemon/               # 포켓몬 도메인 데이터
│           ├── prototypes.npz     # 클래스별 프로토타입 임베딩
│           ├── prototypes.meta.json # 메타데이터 (클래스명, 통계 등)
│           ├── pokemon.names.json # 포켓몬 (한글 - 영어 이름 json)
│           └── *.npz              # 개별 포켓몬 임베딩 (optional)
├── test_images/                   # 테스트용 이미지 샘플
├── requirements.txt               # Python 의존성
├── .env                           # 환경 변수 (git ignore)
├── .env.example                   # 환경 변수 템플릿
└── CLAUDE.md                      # 프로젝트 가이드 (본 문서)
```

## 레이어 아키텍처 설명

### 1. API Layer (`app/api/`)
- **목적**: HTTP 요청/응답 처리, 라우팅
- **역할**:
  - 엔드포인트 정의 및 경로 매핑
  - 요청 검증 (Pydantic)
  - 응답 직렬화
- **원칙**:
  - 비즈니스 로직 포함 금지 (Service Layer에 위임)
  - 버전별 라우터 분리 (`/api/v1`, `/api/v2`)
  - OpenAPI 문서 자동 생성 활용

### 2. Core Layer (`app/core/`)
- **목적**: 애플리케이션 전역 설정 및 의존성 관리
- **구성요소**:
  - `config.py`: Pydantic Settings로 환경 변수 관리
  - `dependencies.py`: FastAPI Depends용 의존성 함수
- **원칙**:
  - 싱글톤 패턴으로 리소스 관리 (ModelManager)
  - 환경 변수 우선순위: `.env` > 기본값

### 3. Models Layer (`app/models/`)
- **목적**: AI/ML 모델 래퍼 및 저수준 연산
- **역할**:
  - PyTorch 모델 로딩 및 추론
  - NumPy 기반 수치 연산
- **원칙**:
  - 모델 초기화는 앱 시작 시 1회 (lifespan)
  - 비즈니스 로직 분리 (Service Layer에서 호출)

### 4. Schemas Layer (`app/schemas/`)
- **목적**: 데이터 검증 및 직렬화
- **역할**:
  - 요청 바디 검증 (Pydantic BaseModel)
  - 응답 모델 정의
  - OpenAPI 스키마 자동 생성
- **원칙**:
  - 모든 API 입출력은 스키마로 정의
  - 예제 데이터 포함 (`Config.json_schema_extra`)

### 5. Services Layer (`app/services/`)
- **목적**: 비즈니스 로직 구현
- **역할**:
  - Models Layer 호출 및 결과 가공
  - 에러 처리 및 검증
- **원칙**:
  - 단일 책임 원칙 (SRP)
  - FastAPI 의존성 주입 활용

### 6. Data Layer (`app/ai_models/`)
- **목적**: AI 모델 데이터 저장소
- **구조**:
  - 도메인별 디렉토리 (`pokemon/`, `animals/`, ...)
  - 프로토타입 임베딩 (NPZ 파일)
  - 메타데이터 (JSON 파일)
- **원칙**:
  - 하드코딩 금지 → 파일 기반 관리
  - 도메인 추가 시 새 디렉토리 생성

---

# FastAPI 개발 가이드

## 1. 프로젝트 초기화

### 가상환경 설정
```bash
# 가상환경 생성
python -m venv .venv

# 가상환경 활성화
# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 환경 변수 설정
```bash
# .env.example을 복사하여 .env 생성
cp .env.example .env

# .env 파일 편집 (필요한 설정값 입력)
# 예시:
# MODEL_NAME=openai/clip-vit-base-patch32
# DEVICE=cuda  # 또는 cpu
# TAU1=0.30
# TAU2=0.04
```

### 개발 서버 실행
```bash
# 방법 1: uvicorn 직접 실행 (권장)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 방법 2: FastAPI CLI 사용
fastapi dev app/main.py

# 방법 3: python 직접 실행
python -m app.main
```

## 2. 새로운 API 엔드포인트 추가

### 단계 1: Schema 정의 (`app/schemas/`)
```python
# app/schemas/new_feature.py
from pydantic import BaseModel, Field

class NewFeatureRequest(BaseModel):
    """요청 스키마"""
    input_data: str = Field(..., description="입력 데이터")

class NewFeatureResponse(BaseModel):
    """응답 스키마"""
    result: str = Field(..., description="처리 결과")

    class Config:
        json_schema_extra = {
            "example": {
                "result": "Success"
            }
        }
```

### 단계 2: Service 구현 (`app/services/`)
```python
# app/services/new_feature_service.py
class NewFeatureService:
    """비즈니스 로직"""

    def __init__(self, dependency):
        self.dependency = dependency

    def process(self, input_data: str) -> dict:
        # 실제 비즈니스 로직 구현
        result = self.dependency.do_something(input_data)
        return {"result": result}
```

### 단계 3: Endpoint 정의 (`app/api/v1/endpoints/`)
```python
# app/api/v1/endpoints/new_feature.py
from fastapi import APIRouter, Depends
from app.schemas.new_feature import NewFeatureRequest, NewFeatureResponse
from app.services.new_feature_service import NewFeatureService

router = APIRouter(prefix="/new-feature", tags=["new-feature"])

@router.post("/process", response_model=NewFeatureResponse)
async def process_new_feature(
    request: NewFeatureRequest,
    dependency=Depends(get_dependency)
):
    """새로운 기능 처리 엔드포인트"""
    service = NewFeatureService(dependency)
    result = service.process(request.input_data)
    return NewFeatureResponse(**result)
```

### 단계 4: Router 등록 (`app/api/v1/router.py`)
```python
# app/api/v1/router.py
from fastapi import APIRouter
from app.api.v1.endpoints import similarity, new_feature

api_router = APIRouter()
api_router.include_router(similarity.router)
api_router.include_router(new_feature.router)  # 추가
```

## 3. 새로운 도메인 데이터 추가

### 단계 1: 데이터 디렉토리 생성
```bash
mkdir -p app/ai_models/animals
```

### 단계 2: 프로토타입 임베딩 생성
```python
# scripts/generate_prototypes.py (예시)
import numpy as np

# 클래스별 임베딩 수집 후 평균 계산
prototypes = {
    "cat": np.mean(cat_embeddings, axis=0),
    "dog": np.mean(dog_embeddings, axis=0),
    # ...
}

# NPZ 파일로 저장
np.savez(
    "app/ai_models/animals/prototypes.npz",
    **prototypes
)
```

### 단계 3: 메타데이터 생성
```json
// app/ai_models/animals/prototypes.meta.json
{
  "domain": "animals",
  "classes": ["cat", "dog", "bird"],
  "model": "openai/clip-vit-base-patch32",
  "created_at": "2025-01-01T00:00:00Z",
  "statistics": {
    "total_classes": 3,
    "embedding_dim": 512
  }
}
```

### 단계 4: Config 업데이트
```python
# app/core/config.py
class Settings(BaseSettings):
    # 도메인별 경로 추가
    animals_prototypes_path: Path = base_dir / "ai_models" / "animals" / "prototypes.npz"
```

## 4. 의존성 주입 패턴

### 싱글톤 패턴 (리소스 관리)
```python
# app/core/dependencies.py
class ModelManager:
    """앱 전역 모델 관리자 (싱글톤)"""
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self):
        if self._model is None:
            self._model = load_heavy_model()

    @property
    def model(self):
        return self._model

model_manager = ModelManager()
```

### FastAPI Depends 활용
```python
# app/core/dependencies.py
def get_model():
    """모델 의존성"""
    return model_manager.model

# app/api/v1/endpoints/similarity.py
@router.post("/predict")
async def predict(
    data: InputData,
    model=Depends(get_model)  # 자동 주입
):
    result = model.predict(data)
    return result
```

## 5. 비동기 처리 가이드

### 언제 async/await를 사용할까?
- **사용해야 할 때**:
  - I/O 바운드 작업 (파일 읽기, 네트워크 요청)
  - 데이터베이스 쿼리 (async driver 사용 시)
  - 외부 API 호출

- **사용하지 말아야 할 때**:
  - CPU 바운드 작업 (ML 추론, 이미지 처리)
  - 동기 라이브러리 사용 시

### 예시
```python
# I/O 바운드: async 사용
@router.post("/from-url")
async def process_url(url: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)  # 비동기 네트워크 요청
    return response.json()

# CPU 바운드: async 불필요 (일반 함수)
@router.post("/inference")
async def inference(image: UploadFile):
    # PyTorch 추론은 CPU 바운드
    # async 함수이지만 내부는 동기 처리
    embedding = model.extract(image)  # CPU 연산
    return {"embedding": embedding}
```

## 6. 에러 처리

### HTTPException 활용
```python
from fastapi import HTTPException, status

@router.post("/process")
async def process(data: InputData):
    try:
        result = service.process(data)
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resource not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}"
        )
```

### 전역 예외 핸들러 (선택)
```python
# app/main.py
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )
```

## 7. 테스트

### 테스트 구조
```
tests/
├── __init__.py
├── conftest.py              # Pytest fixtures
├── test_api/
│   ├── test_similarity.py   # API 엔드포인트 테스트
│   └── test_health.py
├── test_services/
│   └── test_similarity_service.py
└── test_models/
    └── test_embedding_extractor.py
```

### 예시: API 테스트
```python
# tests/test_api/test_similarity.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_find_similar_by_upload():
    with open("test_images/pikachu.jpg", "rb") as f:
        response = client.post(
            "/api/v1/similarity/find-by-upload",
            files={"file": ("pikachu.jpg", f, "image/jpeg")},
            params={"top_k": 3}
        )

    assert response.status_code == 200
    data = response.json()
    assert "verdict" in data
    assert len(data["top_k"]) == 3
```

---

# 개발 원칙 및 Best Practices

## 1. 코드 스타일
- **Formatter**: Black (line length: 100)
- **Linter**: Ruff 또는 Flake8
- **Type Hints**: 모든 함수에 타입 힌트 사용
- **Docstrings**: Google 스타일 또는 NumPy 스타일

```python
def calculate_similarity(
    query_embedding: np.ndarray,
    prototypes: dict[str, np.ndarray],
    top_k: int = 3
) -> dict[str, Any]:
    """
    Calculate cosine similarity between query and prototypes.

    Args:
        query_embedding: Query image embedding vector (shape: [D])
        prototypes: Dictionary of prototype embeddings {class_name: embedding}
        top_k: Number of top results to return

    Returns:
        Dictionary containing top-k results and verdict
    """
    pass
```

## 2. 환경 관리
- **개발/프로덕션 분리**: `.env.dev`, `.env.prod`
- **민감 정보 보호**: `.env`는 `.gitignore`에 추가
- **예시 파일 제공**: `.env.example`로 템플릿 공유

```bash
# .env.example
APP_NAME=Image Similarity API
APP_VERSION=1.0.0
MODEL_NAME=openai/clip-vit-base-patch32
DEVICE=cpu
TAU1=0.30
TAU2=0.04
```

## 3. 성능 최적화
- **모델 로딩**: 앱 시작 시 1회만 (lifespan 이벤트)
- **배치 처리**: 여러 이미지는 배치로 처리
- **캐싱**: 반복 요청은 Redis/메모리 캐시 활용
- **비동기 I/O**: 네트워크 요청은 비동기 처리

```python
# app/main.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 이벤트"""
    # Startup
    print("Loading models...")
    model_manager.initialize()  # 모델 1회 로딩
    print("Ready!")

    yield

    # Shutdown
    print("Cleanup...")
```

## 4. 로깅
```python
import logging

logger = logging.getLogger(__name__)

@router.post("/process")
async def process(data: InputData):
    logger.info(f"Processing request: {data}")
    try:
        result = service.process(data)
        logger.info(f"Success: {result}")
        return result
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise
```

## 5. 보안
- **CORS 설정**: 프로덕션에서는 `allow_origins` 제한
- **파일 업로드 검증**: 파일 크기, 타입 검증
- **Rate Limiting**: 필요시 slowapi 등 사용

```python
# app/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com"],  # 프로덕션
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

---

# 데이터 관리 가이드

## 1. 프로토타입 임베딩 생성 워크플로우

### Step 1: 이미지 수집
```
app/ai_models/pokemon/raw_images/
├── 피카츄/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── 파이리/
│   └── ...
└── ...
```

### Step 2: 임베딩 추출 스크립트
```python
# scripts/generate_prototypes.py
import numpy as np
from pathlib import Path
from app.models.embedding_extractor import ImageEmbeddingExtractor

def generate_prototypes(domain: str):
    extractor = ImageEmbeddingExtractor()
    raw_dir = Path(f"app/ai_models/{domain}/raw_images")
    prototypes = {}

    for class_dir in raw_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        embeddings = []

        for img_path in class_dir.glob("*.jpg"):
            emb = extractor.extract_from_file(str(img_path))
            embeddings.append(emb)

        # 프로토타입 = 평균 임베딩
        prototypes[class_name] = np.mean(embeddings, axis=0)

    # 저장
    output_path = f"app/ai_models/{domain}/prototypes.npz"
    np.savez(output_path, **prototypes)
    print(f"Saved {len(prototypes)} prototypes to {output_path}")

if __name__ == "__main__":
    generate_prototypes("pokemon")
```

### Step 3: 메타데이터 생성
```python
# scripts/generate_metadata.py
import json
from datetime import datetime

metadata = {
    "domain": "pokemon",
    "classes": list(prototypes.keys()),
    "model": "openai/clip-vit-base-patch32",
    "created_at": datetime.now().isoformat(),
    "statistics": {
        "total_classes": len(prototypes),
        "embedding_dim": 512,
        "avg_samples_per_class": 50
    }
}

with open("app/ai_models/pokemon/prototypes.meta.json", "w") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
```

## 2. 새로운 도메인 추가 체크리스트

- [ ] 1. `app/ai_models/{domain}/` 디렉토리 생성
- [ ] 2. 원본 이미지 수집 → `raw_images/` 저장
- [ ] 3. 프로토타입 임베딩 생성 스크립트 실행
- [ ] 4. `prototypes.npz` 파일 생성 확인
- [ ] 5. `prototypes.meta.json` 메타데이터 생성
- [ ] 6. `app/core/config.py`에 경로 추가
- [ ] 7. API 엔드포인트에 도메인 파라미터 추가
- [ ] 8. 테스트 이미지로 검증

---

# API 문서

## 자동 생성 문서
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## 주요 엔드포인트

### 1. Health Check
```http
GET /health
```
**Response**: `{"status": "healthy"}`

### 2. 이미지 업로드 유사도 검색
```http
POST /api/v1/similarity/find-by-upload
Content-Type: multipart/form-data

file=@pikachu.jpg
top_k=3
```

**Response**:
```json
{
  "top_k": [
    ["피카츄", 0.85],
    ["라이츄", 0.72],
    ["파이리", 0.65]
  ],
  "verdict": "피카츄",
  "s1": 0.85,
  "margin": 0.13,
  "is_unknown": false
}
```

### 3. URL 기반 유사도 검색
```http
POST /api/v1/similarity/find-by-url
Content-Type: application/json

{
  "url": "https://example.com/pikachu.jpg"
}
```

---

# 배포 가이드

## 1. Docker 배포 (권장)

### Dockerfile 예시
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 복사
COPY app/ ./app/

# 포트 노출
EXPOSE 8000

# 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME=openai/clip-vit-base-patch32
      - DEVICE=cpu
    volumes:
      - ./app/ai_models:/app/app/ai_models
```

## 2. 시스템 요구사항
- **CPU**: 4코어 이상 (추론용)
- **RAM**: 8GB 이상
- **GPU**: CUDA 지원 GPU (선택, 성능 향상)
- **Storage**: 5GB 이상 (모델 + 데이터)

## 3. 환경별 설정
```bash
# Development
uvicorn app.main:app --reload --env-file .env.dev

# Production
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --env-file .env.prod
```

---

# 트러블슈팅

## 문제 1: 모델 로딩 느림
**원인**: CLIP 모델 다운로드 시간
**해결**:
```python
# Hugging Face 캐시 디렉토리 설정
export HF_HOME=/path/to/cache

# 또는 config.py에서:
os.environ["HF_HOME"] = "/app/cache"
```

## 문제 2: Out of Memory
**원인**: GPU 메모리 부족
**해결**:
```python
# config.py
device = "cpu"  # GPU 대신 CPU 사용

# 또는 배치 크기 줄이기
```

## 문제 3: CORS 에러
**원인**: 프론트엔드 도메인 허용 안됨
**해결**:
```python
# app/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 프론트엔드 URL 추가
    ...
)
```

---

# 향후 개발 로드맵

## Phase 1: 다양한 도메인 지원
- [ ] 동물 도메인 추가
- [ ] 제품 도메인 추가
- [ ] 도메인 동적 선택 API

## Phase 2: 성능 최적화
- [ ] Redis 캐싱 도입
- [ ] 배치 추론 API
- [ ] GPU 최적화 (ONNX, TensorRT)

## Phase 3: 기능 확장
- [ ] 사용자 피드백 수집 API
- [ ] 실시간 스트리밍 (WebSocket)
- [ ] 모델 버전 관리 시스템

## Phase 4: 모니터링
- [ ] Prometheus + Grafana 연동
- [ ] 로그 수집 (ELK Stack)
- [ ] 알림 시스템 (Slack, Email)

---

# 참고 자료

## FastAPI 공식 문서
- [FastAPI 공식 사이트](https://fastapi.tiangolo.com/)
- [Pydantic 문서](https://docs.pydantic.dev/)
- [Uvicorn 문서](https://www.uvicorn.org/)

## 프로젝트 관련
- **CLIP 모델**: [OpenAI CLIP](https://github.com/openai/CLIP)
- **Hugging Face Transformers**: [Docs](https://huggingface.co/docs/transformers)
- **NumPy**: [Docs](https://numpy.org/doc/)

## 코드 스타일 가이드
- [PEP 8](https://peps.python.org/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

---

# 라이선스
MIT License (프로젝트에 맞게 수정)
