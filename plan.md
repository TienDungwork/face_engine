# Plan: Implement `/searchEvent` in `face-engine` (LPB DB style)

## Objective
Add `/api/v1/searchEvent` to `face-engine` with behavior aligned to `lpb-face-cpu`, using MSSQL event search with LPB-style table/columns.

## Scope Decisions
- Input image source: **base64 only** (`img_base64`).
- `img_url`: ignored in service logic.
- Event DB mapping: use LPB-style event table and columns.

## Implementation Plan

### Step 1: Add Event Schemas
Create `app/schemas/analyze/search_event.py` with:
- `EventSearchRequest`
  - `img_base64`
  - `comIds`
  - `threshold`
  - `quality`
  - `num_result`
  - `fromDate`
  - `toDate`
- `EventResult` (`eventId`, `score`)
- `EventSearchResponse` (`timestamp`, `status`, `error`, `data`)

### Step 2: Implement Event Service (Base64 Only)
Create `app/services/analyze/search_event.py`:
- Validate and decode `img_base64`.
- Detect faces via InsightFace.
- Select largest face and extract embedding.
- Detect mask from cropped face.
- Call `database_manager.search_event(...)`.
- Return standardized success/error response.

### Step 3: Add Event Search in Database Manager
Update `app/core/database_manager.py`:
- Add method `search_event(face_embedding, com_ids, threshold, num_result, from_date, to_date, wear_mask)`.
- Query MSSQL directly for events (not local SQLite cache):
  - filter by company IDs
  - filter by access time range
  - require non-null feature
- Decode event feature embeddings.
- Compute cosine similarity, adjust threshold using mask signal.
- Sort by score desc and return top `num_result` as `EventResult`.

### Step 4: Update Environment and Config for LPB Event Mapping
Update `resources/.env`:
- `EVENT_TABLE_NAME="EventFeatures"`
- Add event column variables:
  - `EVENT_ID_COLUMN="EventId"`
  - `EVENT_FEATURE_COLUMN="FeaturePath"`
  - `EVENT_COMPID_COLUMN="CompId"`
  - `EVENT_TIME_COLUMN="AccessTime"`

Update `app/core/config.py`:
- Add new `AppConfig` fields for the event column names.
- Ensure `search_event` reads table/column names from config.

### Step 5: Add API Endpoint and Router Registration
Create `app/api/v1/endpoints/analyze/search_event.py`:
- `POST /searchEvent` with request/response models.

Update `app/api/v1/endpoints/analyze/__init__.py`:
- include `search_event_router`.

### Step 6: Verification
Run manual API checks for `/api/v1/searchEvent`:
- Valid base64 face image.
- Invalid base64 payload.
- No-face image.
- Company/date filters.
- Threshold effect with/without mask.

## Files To Change
- `app/schemas/analyze/search_event.py` (new)
- `app/services/analyze/search_event.py` (new)
- `app/core/database_manager.py`
- `app/core/config.py`
- `app/api/v1/endpoints/analyze/search_event.py` (new)
- `app/api/v1/endpoints/analyze/__init__.py`
- `resources/.env`
