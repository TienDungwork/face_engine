import base64
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from app.core.config import app_config, get_smart_face_dsn
from app.core.smart_face_pool import (
    get_last_smart_face_pool_error,
    get_smart_face_pool,
    init_smart_face_pool,
    validate_events_table_ref,
    validate_qualified_table_name,
)
from app.schemas.analyze.search_face import FaceSearchRequest, FaceSearchResponse
from app.schemas.analyze.search_face_events import (
    FaceEventSearchItem,
    SearchFaceEventsRequest,
    SearchFaceEventsResponse,
)
from app.services.analyze.search_face import _detect_faces, _get_valid_image
from app.utils.helpers import (
    batch_cosine_similarity,
    decode_smf_face_feature_bytes,
    encode_embedding,
)


def _format_image_field(val: Any) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, str):
        return val
    if isinstance(val, (bytes, memoryview)):
        b = bytes(val)
        try:
            return b.decode("utf-8")
        except UnicodeDecodeError:
            return base64.b64encode(b).decode("ascii")
    return str(val)


def _to_utc_aware(dt: datetime, assume_tz: str) -> datetime:
    """
    Tránh lỗi trộn naive/aware: mọi giá trị đưa về UTC có timezone.

    Quy ước để "time window" không bị lệch:
    - Nếu client gửi datetime có tzinfo (vd: "...Z" hoặc "+07:00") => tin vào offset.
    - Nếu client gửi datetime KHÔNG có tzinfo (naive) => coi như giờ local theo `assume_tz`
      (múi giờ của access_time trong PG), rồi convert sang UTC.
    """
    if dt.tzinfo is None:
        try:
            tz = ZoneInfo(assume_tz)
        except Exception:
            tz = timezone.utc
        return dt.replace(tzinfo=tz).astimezone(timezone.utc)
    return dt.astimezone(timezone.utc)


def _id_filter_text(v: Union[int, str]) -> str:
    """Lọc UUID / bigint / varchar / enum: so sánh qua text trên Postgres."""
    return str(v).strip()


def _sql_access_time_zone() -> str:
    """Tên zone cho PG AT TIME ZONE (an toàn ghép vào SQL). Mặc định UTC."""
    raw = (app_config.SMART_FACE_ACCESS_TIME_TZ or "").strip() or "UTC"
    if not re.fullmatch(r"[A-Za-z0-9_/+\-]+", raw):
        print(f"[smart_face] SMART_FACE_ACCESS_TIME_TZ không hợp lệ: {raw!r}, dùng UTC")
        return "UTC"
    return raw


def _should_apply_id_filter(v: Optional[Union[int, str]]) -> bool:
    """
    Swagger/Wrappler hay gửi 0 cho mọi field optional.
    None, '', '0', 0 → không thêm điều kiện SQL (camera_id UUID / direction varchar sẽ không khớp '0').
    """
    if v is None:
        return False
    s = str(v).strip()
    return s not in ("", "0")


def _resolve_time_range(
    from_time: Optional[datetime],
    to_time: Optional[datetime],
    assume_tz: str,
) -> Tuple[datetime, datetime]:
    now = datetime.now(timezone.utc)
    if to_time is None:
        to_t = now
    else:
        to_t = _to_utc_aware(to_time, assume_tz)
    if from_time is None:
        days = app_config.SMART_FACE_DEFAULT_SEARCH_DAYS
        from_t = to_t - timedelta(days=days)
    else:
        from_t = _to_utc_aware(from_time, assume_tz)
    # from_time == to_time (FE gửi trùng) => khoảng 0s, luôn 0 dòng — lùi from theo N ngày
    if from_t == to_t:
        from_t = to_t - timedelta(days=app_config.SMART_FACE_DEFAULT_SEARCH_DAYS)
    return from_t, to_t


def _build_select_sql(
    events_table: str,
    with_camera_join: bool,
    camera_fqn: str,
    access_tz: str,
) -> str:
    cam_code = "c.code AS camera_code" if with_camera_join else "NULL::text AS camera_code"
    cam_name = "c.name AS camera_name" if with_camera_join else "NULL::text AS camera_name"
    join = ""
    if with_camera_join:
        join = f" LEFT JOIN {camera_fqn} c ON c.id = e.camera_id "
    # access_time = timestamp without time zone: hiểu là "giờ tường" trong access_tz.
    # So với from_time/to_time (UTC) bind timestamptz — Postgres tự đối chiếu đúng instant.
    return f"""
SELECT
    e.id,
    e.event_id,
    e.access_time,
    e.camera_id,
    {cam_code},
    {cam_name},
    e.user_code,
    e.user_name,
    e.department_name,
    e.image,
    e.direction,
    e.score_match,
    e.face_feature
FROM {events_table} e
{join}
WHERE e.face_feature IS NOT NULL
  AND (e.access_time AT TIME ZONE '{access_tz}') >= $1::timestamptz
  AND (e.access_time AT TIME ZONE '{access_tz}') <= $2::timestamptz
"""


async def search_face_events(
    request: SearchFaceEventsRequest,
) -> SearchFaceEventsResponse:
    ts = datetime.now().isoformat()
    dsn = (get_smart_face_dsn() or "").strip()
    pool = get_smart_face_pool()
    if pool is None and dsn:
        await init_smart_face_pool()
        pool = get_smart_face_pool()

    if pool is None:
        if not dsn:
            err = (
                "Thiếu cấu hình kết nối smart_face: đặt SMART_FACE_DATABASE_URL "
                "hoặc đủ bộ SMART_FACE_DB_HOST, SMART_FACE_DB_USER, SMART_FACE_DB_NAME "
                "(và SMART_FACE_DB_PASSWORD nếu có) trong resources/.env, rồi khởi động lại service."
            )
        else:
            detail = (get_last_smart_face_pool_error() or "").strip()
            hint = (
                " Kiểm tra: mật khẩu postgres, pg_hba cho phép IP của container, firewall, "
                "và thử SMART_FACE_DB_SSL=disable trong .env nếu server PG không dùng SSL."
            )
            if detail:
                err = f"Không kết nối được PostgreSQL smart_face: {detail}.{hint}"
            else:
                err = (
                    "Pool smart_face chưa tạo được (lỗi không ghi lại được)."
                    + hint
                )
        return SearchFaceEventsResponse(
            timestamp=ts,
            status=503,
            error=err,
            data=None,
        )

    access_tz = _sql_access_time_zone()
    from_t, to_t = _resolve_time_range(request.from_time, request.to_time, access_tz)
    if from_t > to_t:
        return SearchFaceEventsResponse(
            timestamp=ts,
            status=400,
            error="from_time phải nhỏ hơn hoặc bằng to_time",
            data=None,
        )

    limit_cap = min(request.limit, app_config.SMART_FACE_EVENTS_MAX_ROWS)

    image = await _get_valid_image(
        FaceSearchRequest(
            img_base64=request.img_base64,
            img_url=request.img_url,
        )
    )
    if isinstance(image, FaceSearchResponse):
        return SearchFaceEventsResponse(
            timestamp=image.timestamp,
            status=image.status,
            error=image.error,
            data=image.data,
        )

    faces = await _detect_faces(image)
    if isinstance(faces, FaceSearchResponse):
        return SearchFaceEventsResponse(
            timestamp=faces.timestamp,
            status=faces.status,
            error=faces.error,
            data=faces.data,
        )

    largest = max(
        faces,
        key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
    )
    query_vec = largest.embedding.astype(np.float32, copy=False)
    if query_vec.shape[0] != 512:
        return SearchFaceEventsResponse(
            timestamp=ts,
            status=400,
            error="Vector khuôn mặt không đúng 512 chiều",
            data=None,
        )

    events_table = (app_config.SMART_FACE_EVENTS_TABLE or "smf_face_events").strip()
    if not validate_events_table_ref(events_table):
        return SearchFaceEventsResponse(
            timestamp=ts,
            status=400,
            error="SMART_FACE_EVENTS_TABLE không hợp lệ (vd: smf_face_events hoặc public.smf_face_events)",
            data=None,
        )

    camera_fqn = (
        app_config.SMART_FACE_CAMERA_TABLE_QUALIFIED
        or app_config.SMART_FACE_CAMERA_TABLE_FQN
        or ""
    ).strip()
    with_cam = bool(camera_fqn and validate_qualified_table_name(camera_fqn))
    if camera_fqn and not with_cam:
        return SearchFaceEventsResponse(
            timestamp=ts,
            status=400,
            error=(
                "SMART_FACE_CAMERA_TABLE_QUALIFIED / SMART_FACE_CAMERA_TABLE_FQN không hợp lệ "
                "(vd: vms_db.camera — chỉ chữ, số, gạch dưới và dấu chấm)"
            ),
            data=None,
        )

    base_sql = _build_select_sql(events_table, with_cam, camera_fqn, access_tz)
    conds: List[str] = []
    # from_t/to_t đã được chuẩn hoá về UTC-aware trong _resolve_time_range()
    params: List[Any] = [from_t, to_t]
    idx = 3

    if _should_apply_id_filter(request.camera_id):
        conds.append(f"e.camera_id::text = ${idx}::text")
        params.append(_id_filter_text(request.camera_id))
        idx += 1
    if _should_apply_id_filter(request.company_id):
        conds.append(f"e.company_id::text = ${idx}::text")
        params.append(_id_filter_text(request.company_id))
        idx += 1
    if _should_apply_id_filter(request.department_id):
        conds.append(f"e.department_id::text = ${idx}::text")
        params.append(_id_filter_text(request.department_id))
        idx += 1
    if _should_apply_id_filter(request.direction):
        conds.append(f"e.direction::text = ${idx}::text")
        params.append(_id_filter_text(request.direction))
        idx += 1

    where_extra = ""
    if conds:
        where_extra = " AND " + " AND ".join(conds)

    # LIMIT không bind tham số: tránh lệch placeholder + ép int an toàn (1..max)
    max_rows = int(app_config.SMART_FACE_EVENTS_MAX_ROWS)
    safe_limit = max(1, min(int(limit_cap), max_rows))
    sql = f"{base_sql}{where_extra} ORDER BY e.access_time DESC LIMIT {safe_limit}"

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
    except Exception as e:
        msg = str(e)
        hint = ""
        if "does not exist" in msg and with_cam:
            hint = (
                " Bảng camera trong SMART_FACE_CAMERA_TABLE_QUALIFIED không tồn tại trong DB này — "
                "sửa đúng schema.tên_bảng (trong cùng DB smart_face) hoặc xóa/comment biến đó để không JOIN."
            )
        return SearchFaceEventsResponse(
            timestamp=datetime.now().isoformat(),
            status=500,
            error=f"Lỗi truy vấn {events_table}: {msg}.{hint}",
            data=None,
        )

    table_has_face_feature: Optional[bool] = None
    if len(rows) == 0:
        try:
            async with pool.acquire() as conn:
                table_has_face_feature = await conn.fetchval(
                    f"SELECT EXISTS (SELECT 1 FROM {events_table} e "
                    f"WHERE e.face_feature IS NOT NULL)"
                )
        except Exception:
            table_has_face_feature = None

    embeddings: List[np.ndarray] = []
    metas: List[Dict[str, Any]] = []
    rows_decode_failed = 0
    for row in rows:
        ff = row["face_feature"]
        if ff is None:
            rows_decode_failed += 1
            continue
        b = bytes(ff) if not isinstance(ff, bytes) else ff
        try:
            emb = decode_smf_face_feature_bytes(b)
        except Exception:
            rows_decode_failed += 1
            continue
        if emb.shape[0] != 512:
            rows_decode_failed += 1
            continue
        embeddings.append(emb)
        metas.append(dict(row))

    def _input_extra() -> Dict[str, Any]:
        ex: Dict[str, Any] = {
            "access_time_at_zone": access_tz,
            "time_bounds_utc": {
                "from": from_t.isoformat(),
                "to": to_t.isoformat(),
            },
        }
        if table_has_face_feature is not None:
            ex["table_has_any_face_feature"] = table_has_face_feature
        if len(rows) > 0 and not embeddings:
            ex["rows_decode_failed"] = rows_decode_failed
        return ex

    if not embeddings:
        hint = ""
        if len(rows) == 0 and table_has_face_feature is True:
            hint = (
                " Có face_feature trong bảng nhưng 0 dòng sau lọc — thường do: (1) khoảng thời gian UTC "
                "không chứa access_time; (2) Swagger gửi camera_id/company_id/department_id = 0 — đã bỏ qua 0; "
                "(3) SMART_FACE_ACCESS_TIME_TZ (access_time_at_zone trong response) phải khớp cách DB lưu timestamp."
            )
        elif len(rows) == 0 and table_has_face_feature is False:
            hint = " Không có dòng nào có face_feature khác NULL — kiểm tra bảng/schema đúng DB."
        elif len(rows) > 0:
            hint = (
                " Có dòng trong khoảng thời gian nhưng không đọc được vector 512 chiều từ face_feature "
                "(BYTEA base64 / 2048 byte float32 / SECRET_KEY)."
            )
        return SearchFaceEventsResponse(
            timestamp=datetime.now().isoformat(),
            status=200,
            error="Success",
            data={
                "events": [],
                "input_data": {
                    "featureBase64": encode_embedding(query_vec),
                    "from_time": from_t.isoformat(),
                    "to_time": to_t.isoformat(),
                    "threshold": request.threshold,
                    "rows_scanned": len(rows),
                    "hint": hint.strip() or None,
                    **_input_extra(),
                },
            },
        )

    mat = np.stack(embeddings, axis=0).astype(np.float32, copy=False)
    sims = batch_cosine_similarity(query_vec, mat)
    # Đồng bộ semantics với search_person/searchFace:
    # - search_face chia threshold /= 2 để so với cosine thô
    # - rồi khi trả về, nhân scoreMatching * 2 (cap 1)
    # => search_face_events cần dùng threshold/2 khi filter, và trả similarity theo score scaled.
    effective_threshold = float(request.threshold) / 2.0
    best_idx = int(np.argmax(sims)) if len(sims) > 0 else -1
    best_sim_raw = float(sims[best_idx]) if best_idx >= 0 else None
    best_sim = float(min(best_sim_raw * 2.0, 1.0)) if best_sim_raw is not None else None
    best_event_id = str(metas[best_idx].get("id")) if best_idx >= 0 and metas else None

    items: List[FaceEventSearchItem] = []
    for i, sim in enumerate(sims):
        sim_raw = float(sim)
        if sim_raw < effective_threshold:
            continue
        sim_scaled = float(min(sim_raw * 2.0, 1.0))
        m = metas[i]
        at = m.get("access_time")
        at_str = at.isoformat() if isinstance(at, datetime) else (
            str(at) if at is not None else None
        )
        sm = m.get("score_match")
        score_f: Optional[float] = None
        if sm is not None:
            try:
                score_f = float(sm)
            except (TypeError, ValueError):
                score_f = None
        ev_id = m.get("event_id")
        ev_str: Optional[str] = None
        if ev_id is not None:
            ev_str = str(ev_id)
        cam_raw = m.get("camera_id")
        cam_str: Optional[str] = (
            str(cam_raw) if cam_raw is not None else None
        )
        dir_raw = m.get("direction")
        dir_str: Optional[str] = (
            str(dir_raw) if dir_raw is not None else None
        )
        items.append(
            FaceEventSearchItem(
                id=int(m["id"]),
                event_id=ev_str,
                access_time=at_str,
                camera_id=cam_str,
                camera_code=m.get("camera_code"),
                camera_name=m.get("camera_name"),
                user_code=m.get("user_code"),
                user_name=m.get("user_name"),
                department_name=m.get("department_name"),
                image=_format_image_field(m.get("image")),
                direction=dir_str,
                score_match=score_f,
                similarity=sim_scaled,
            )
        )

    items.sort(key=lambda x: x.similarity, reverse=True)

    return SearchFaceEventsResponse(
        timestamp=datetime.now().isoformat(),
        status=200,
        error="Success",
        data={
            "events": [it.model_dump() for it in items],
            "input_data": {
                "featureBase64": encode_embedding(query_vec),
                "from_time": from_t.isoformat(),
                "to_time": to_t.isoformat(),
                "threshold": request.threshold,
                "rows_scanned": len(rows),
                **_input_extra(),
                "best_similarity": best_sim,
                "best_event_id": best_event_id,
            },
        },
    )
