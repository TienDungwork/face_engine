#!/usr/bin/env python3
"""
Bắn lịch sử nhận diện (face event) lên MQTT – dùng 1 nhân viên thật đã có trong DB smart_face.
Backend nhận topic smart_vms/ai_events/FACE, route vào handle_face_event và ghi SMF_FaceEvents.

Cách chạy (từ thư mục gốc project, đã cài deps backend):
  cd backend && python -c "import sys; sys.path.insert(0, '..'); exec(open('../scripts/publish_face_history_event.py').read())"
  # hoặc từ project root (nếu PYTHONPATH có backend):
  python scripts/publish_face_history_event.py [--code NV001] [--camera-id UUID] [--direction IN|OUT] [--count 1]

Cấu hình: .env hoặc backend/.env (SMART_FACE_DB_*, MQTT_HOST, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD).
DEMO_CAMERA_ID hoặc --camera-id: UUID camera trong vms_db.camera (nếu không có có thể để trống, event vẫn lưu được).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Load .env từ project root hoặc backend
_project_root = Path(__file__).resolve().parent.parent
for _env in (_project_root / "backend" / ".env", _project_root / ".env"):
    if _env.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(_env)
            break
        except Exception:
            pass

import paho.mqtt.client as mqtt

# Thêm backend vào path để dùng db/settings (nếu chạy từ project root)
_backend = _project_root / "backend"
if _backend.exists() and str(_backend) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    import asyncpg
except ImportError:
    print("Cần cài asyncpg (chạy từ backend: pip install -r backend/requirements.txt)")
    sys.exit(1)


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


async def get_employee_from_db(code: str | None = None) -> dict | None:
    """Lấy 1 nhân viên thật từ smart_face: Code, Fullname, dep_name (từ SMF_Departments)."""
    host = _env("SMART_FACE_DB_HOST", "192.168.1.215")
    port = int(_env("SMART_FACE_DB_PORT", "5432"))
    user = _env("SMART_FACE_DB_USER", "postgres")
    password = _env("SMART_FACE_DB_PASSWORD", "Atin@123#")
    dbname = _env("SMART_FACE_DB_NAME", "smart_face")

    conn = await asyncpg.connect(
        host=host, port=port, user=user, password=password, database=dbname
    )
    try:
        if code:
            row = await conn.fetchrow(
                '''
                SELECT e."Code", e."Fullname", d."Name" AS dep_name
                FROM "SMF_Employees" e
                LEFT JOIN "SMF_Departments" d ON e."DepId" = d."Id"
                WHERE e."IsDelete" = FALSE AND e."Code" = $1
                ''',
                code,
            )
        else:
            row = await conn.fetchrow(
                '''
                SELECT e."Code", e."Fullname", d."Name" AS dep_name
                FROM "SMF_Employees" e
                LEFT JOIN "SMF_Departments" d ON e."DepId" = d."Id"
                WHERE e."IsDelete" = FALSE
                ORDER BY e."Code"
                LIMIT 1
                '''
            )
        if not row:
            return None
        return {
            "user_code": row["Code"],
            "user_name": row["Fullname"] or "",
            "dep_name": row["dep_name"] or "",
        }
    finally:
        await conn.close()


def publish_face_events(
    employee: dict,
    camera_id: str | None,
    direction: str,
    count: int,
) -> None:
    """Publish count lần face event lên MQTT (topic smart_vms/ai_events/FACE)."""
    broker = _env("MQTT_HOST", "192.168.1.215")
    port = int(_env("MQTT_PORT", "1883"))
    username = _env("MQTT_USERNAME", "atin")
    password = _env("MQTT_PASSWORD", "team1@123#")
    topic = "smart_vms/ai_events/FACE"

    client = mqtt.Client()
    if username:
        client.username_pw_set(username, password)
    client.connect(broker, port, 60)
    client.loop_start()

    for i in range(count):
        payload = {
            "ai_modules": ["FACE"],
            "camera_id": camera_id,
            "access_time": datetime.now(timezone.utc).isoformat(),
            "user_code": employee["user_code"],
            "user_name": employee["user_name"],
            "dep_name": employee["dep_name"],
            "direction": direction.upper() if direction.upper() in ("IN", "OUT") else "IN",
        }
        body = json.dumps(payload, ensure_ascii=False)
        result = client.publish(topic, body, qos=1, retain=False)
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"  [{i+1}/{count}] Đã gửi FACE event: user_code={employee['user_code']} direction={payload['direction']}")
        else:
            print(f"  [{i+1}/{count}] Lỗi gửi: rc={result.rc}")
    client.loop_stop()
    client.disconnect()
    print("✅ Kết thúc. Kiểm tra log backend: [MQTT] FACE event on ... / FACE event saved to DB.")


def main():
    parser = argparse.ArgumentParser(description="Bắn face event (nhân viên thật từ DB) lên MQTT để test.")
    parser.add_argument("--code", type=str, default=None, help="Mã nhân viên (Code). Nếu không truyền thì lấy nhân viên đầu tiên.")
    parser.add_argument("--camera-id", type=str, default=None, help="UUID camera (vms_db.camera). Mặc định lấy từ env DEMO_CAMERA_ID.")
    parser.add_argument("--direction", type=str, default="IN", choices=["IN", "OUT"], help="Hướng: IN hoặc OUT.")
    parser.add_argument("--count", type=int, default=1, help="Số event gửi liên tiếp (mặc định 1).")
    args = parser.parse_args()

    camera_id = args.camera_id or _env("DEMO_CAMERA_ID", "d4377cc9-0e86-47d4-9e50-e0a7caa58d7b") or None
    if camera_id == "":
        camera_id = None

    print("Đang lấy 1 nhân viên từ DB smart_face...")
    employee = asyncio.run(get_employee_from_db(args.code))
    if not employee:
        print("❌ Không tìm thấy nhân viên nào (hoặc --code không tồn tại). Kiểm tra DB SMF_Employees.")
        sys.exit(1)
    print(f"  Nhân viên: {employee['user_code']} - {employee['user_name']} - {employee['dep_name']}")

    print(f"\nĐang gửi {args.count} face event lên MQTT (direction={args.direction}, camera_id={camera_id})...")
    publish_face_events(employee, camera_id, args.direction, args.count)


if __name__ == "__main__":
    main()
