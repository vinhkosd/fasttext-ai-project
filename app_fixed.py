import fasttext  # type: ignore
import re
import requests     # type: ignore
from datetime import datetime, timezone, timedelta
from typing import Optional
from flask import Flask, request, jsonify
from dateutil import parser # type: ignore
from dateutil import parser, tz  # type: ignore

app = Flask(__name__)

# ✅ Dùng 1 list duy nhất chứa cả DATA_NAME và DATA_KEY
keys_list = []

# Load model
model = fasttext.load_model('models/intent_model.bin')
DUCKLING_URL = "http://localhost:8085/parse"
VI_LOCALE = "vi_VN"
TZ = timezone(timedelta(hours=7))  # Asia/Ho_Chi_Minh

# =============================
# ✅ xử lý - / trong date
# =============================
def preprocess_date_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\bthang\b", "tháng", text)
    text = re.sub(r"\bnam\b", "năm", text)
    text = re.sub(r"[-\.]", "/", text)
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r'\b(\d{1,2})[/-](\d{2,4})\b', r'tháng \1 năm \2', text)
    text = re.sub(r'\bt\s*0*(\d{1,2})[/-](\d{2})\b', r'tháng \1 năm 20\2', text)
    return text

# =============================
# ✅ Hàm chuyển múi giờ
# =============================
def to_vn_timezone(dt_str: str):
    try:
        dt = parser.isoparse(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        vn_tz = tz.gettz("Asia/Ho_Chi_Minh")
        vn_time = dt.astimezone(vn_tz)

        diff_hours = (vn_tz.utcoffset(dt) - dt.utcoffset()).total_seconds() / 3600
        if diff_hours > 8:
            vn_time += timedelta(days=1)
            vn_time = vn_time.replace(hour=0, minute=0, second=0, microsecond=0)
        return vn_time
    except Exception as e:
        print(f"⚠️ to_vn_timezone error: {e}")
        return parser.isoparse(dt_str)

# =============================
# Gọi Duckling
# =============================
def duckling_parse_time(text: str, ref_time: Optional[datetime] = None):
    text = preprocess_date_text(text)
    if ref_time is None:
        ref_time = datetime.now(TZ)
    print("Duckling đang xử lí")
    reftime_ms = int(ref_time.timestamp() * 1000)

    data = {
        "locale": VI_LOCALE,
        "text": text,
        "dims": '["time"]',
        "reftime": str(reftime_ms),
    }
    try:
        r = requests.post(
            DUCKLING_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"},
            timeout=5
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("⚠️ Duckling error:", e)
        return []

# =============================
# Các hàm hỗ trợ xử lý thời gian
# =============================
def _to_iso(dt: datetime) -> str:
    return dt.isoformat()

def _add_months(dt: datetime, months: int) -> datetime:
    y = dt.year + (dt.month - 1 + months) // 12
    m = (dt.month - 1 + months) % 12 + 1
    return dt.replace(year=y, month=m, day=1, hour=0, minute=0, second=0, microsecond=0)

def _end_of_month(dt: datetime) -> datetime:
    first_next = _add_months(dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0), 1)
    return first_next - timedelta(seconds=1)

# =============================
# Mở rộng khoảng thời gian
# =============================
def _expand_grain_interval(val_iso: str, grain: str, inclusive_end: bool = True, tz: timezone = TZ):
    base = to_vn_timezone(val_iso)
    if base.hour < 3:
        base = base + timedelta(days=1)
    if grain == "day":
        start = base.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        if inclusive_end:
            end = end - timedelta(seconds=1)
        return _to_iso(start), _to_iso(end)
    if grain == "week":
        start = base.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=7)
        if inclusive_end:
            end = end - timedelta(seconds=1)
        return _to_iso(start), _to_iso(end)
    if grain == "month":
        start = base.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_dt = _end_of_month(start) if inclusive_end else _add_months(start, 1)
        return _to_iso(start), _to_iso(end_dt)
    return val_iso, val_iso

# =============================
# Chuẩn hóa dữ liệu từ Duckling
# =============================
def normalize_duckling_times(resp: list, inclusive_end: bool = True, tz: timezone = TZ):
    if not resp:
        return {"type": "none"}
    item = next((x for x in resp if x.get("dim") == "time"), resp[0])
    primary = item.get("value") or item.get("values", [{}])[0]
    if not isinstance(primary, dict):
        return {"type": "none"}
    typ = primary.get("type")
    grain = primary.get("grain")
    if typ == "value":
        val_iso = primary.get("value")
        vn_dt = to_vn_timezone(val_iso)
        val_iso = vn_dt.isoformat()
        if grain in ("week", "month", "quarter", "year"):
            start, end = _expand_grain_interval(val_iso, grain, inclusive_end=inclusive_end, tz=tz)
            return {"type": "range", "start": start, "end": end, "grain": grain}
        return {"type": "single", "date": val_iso, "grain": grain}
    return {"type": "none"}

# =============================
# Intent handling
# =============================
def build_response_with_time(text: str):
    text_lower = text.lower()

    matched_intent = None

    # 🔍 So khớp trực tiếp với keys_list (do frontend push vào)
    for item in keys_list:
        if item.get("DATA_NAME", "").lower() in text_lower:
            matched_intent = item.get("DATA_KEY")
            break

    if not matched_intent:
        return {
            "intent": "NOT_FOUND",
            "confidence": 0.0,
            "time": {"type": "none"},
            "message": "Không tìm thấy intent phù hợp trong danh sách keys_list!"
        }

    # 🕒 Nếu có từ khóa liên quan thời gian thì gọi Duckling
    time_info = {"type": "none"}
    if any(word in matched_intent.upper() for word in ["NGAY", "THANG", "LUONG"]):
        duck_resp = duckling_parse_time(text)
        print(duck_resp)
        time_info = normalize_duckling_times(duck_resp)

    return {
        "intent": matched_intent,
        "confidence": 0.95,
        "time": time_info,
        "message": f"✅ Đã nhận diện intent {matched_intent}"
    }

# =============================
# API routes
# =============================
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    return response

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    print("=== 🧩 DEBUG keys_list ===")
    print(keys_list)
    res = build_response_with_time(text)
    print("=== 🧩 DEBUG BOT RESPONSE ===")
    print(res)
    return jsonify(res)

@app.route('/pushKey', methods=['POST'])
def push_key():
    try:
        data = request.get_json()
        data_name = data.get("DATA_NAME")
        data_key = data.get("DATA_KEY")

        if not data_name or not data_key:
            return jsonify({"error": "Thiếu DATA_NAME hoặc DATA_KEY"}), 400

        keys_list.append({"DATA_NAME": data_name, "DATA_KEY": data_key})

        print(f"📥 Nhận: {data_name} → {data_key}")
        print(f"📦 Tất cả keys_list: {keys_list}")

        return jsonify({
            "message": "Đã thêm dữ liệu thành công!",
            "keys_list": keys_list
        }), 200
    except Exception as e:
        print("❌ Lỗi:", e)
        return jsonify({"error": str(e)}), 500

@app.route('/getKeys', methods=['GET'])
def get_keys():
    try:
        return jsonify({"keys_list": keys_list}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
