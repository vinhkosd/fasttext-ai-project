import fasttext  # type: ignore
import re
import requests     # type: ignore
from datetime import datetime, timezone, timedelta
from typing import Optional
from flask import Flask, request, jsonify
from dateutil import parser # type: ignore
from dateutil import parser, tz  # type: ignore

app = Flask(__name__)

keys_list = []
patterns_list = []

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

    # ✅ Bổ sung: nếu có dạng "1/10" hoặc "12/9" nhưng KHÔNG có chữ 'tháng' hay 'năm' => thêm vào
    # ví dụ "xem lương ngày 1/10" -> "xem lương ngày 1 tháng 10"
    def repl_short_date(m):
        d, mth = int(m.group(1)), int(m.group(2))
        return f"ngày {d} tháng {mth}"

    if not re.search(r"\btháng\b", text):
        text = re.sub(r"\b(\d{1,2})/(\d{1,2})\b", repl_short_date, text)

    # ✅ Ép kiểu dd/mm/yyyy thành "ngày dd tháng mm năm yyyy"
    text = re.sub(
        r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b',
        lambda m: f"ngày {int(m.group(1))} tháng {int(m.group(2))} năm {m.group(3)}",
        text
    )

    # Nếu chỉ có tháng/năm thì đổi
    if not re.search(r'\d{1,2}/\d{1,2}/\d{2,4}', text):
        text = re.sub(r'\b(\d{1,2})[/-](\d{2,4})\b', r'tháng \1 năm \2', text)

    text = re.sub(r'\bt\s*0*(\d{1,2})[/-](\d{2})\b', r'tháng \1 năm 20\2', text)
    return text


# =============================
# Gọi Duckling
# =============================
def duckling_parse_time(text: str, ref_time: Optional[datetime] = None):
    text = preprocess_date_text(text)
    """Gọi Duckling server để parse ngày/giờ."""
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
def _iso_to_dt(s: str) -> datetime:
    s = s.replace('Z', '+00:00')
    return datetime.fromisoformat(s)

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
# HÀM MỞ RỘNG KHOẢNG THỜI GIAN
# =============================
# =============================
# HÀM CHUYỂN MÚI GIỜ CHUẨN
# =============================
def to_vn_timezone(dt_str: str):
    """Chuyển ISO datetime string về múi giờ Việt Nam (UTC+7)."""
    try:
        dt = parser.isoparse(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        vn_tz = tz.gettz("Asia/Ho_Chi_Minh")
        vn_time = dt.astimezone(vn_tz)
        return vn_time
    except Exception as e:
        print(f"⚠️ to_vn_timezone error: {e}")
        return parser.isoparse(dt_str)


# =============================
# HÀM MỞ RỘNG KHOẢNG THỜI GIAN (đã sửa)
# =============================
def _expand_grain_interval(val_iso: str, grain: str, inclusive_end: bool = True, tz: timezone = TZ):
    base = to_vn_timezone(val_iso)

    # ❌ Bỏ logic trừ 1 ngày vì Duckling giờ trả đúng múi giờ VN
    # ⚡ Nếu cần, bạn chỉ mở lại khi Duckling server bị lệch UTC
    # if grain == "day" and base.hour < 3:
    #     base = base - timedelta(days=1)

    if grain == "day":
        start = base.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        if inclusive_end:
            end -= timedelta(seconds=1)
        return _to_iso(start), _to_iso(end)

    if grain == "week":
        start = base.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=7)
        if inclusive_end:
            end -= timedelta(seconds=1)
        return _to_iso(start), _to_iso(end)

    if grain == "month":
        start = base.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_dt = _end_of_month(start) if inclusive_end else _add_months(start, 1)
        return _to_iso(start), _to_iso(end_dt)

    if grain == "quarter":
        q = (base.month - 1) // 3
        start_month = q * 3 + 1
        start = base.replace(month=start_month, day=1, hour=0, minute=0, second=0, microsecond=0)
        end_dt = _add_months(start, 3) - timedelta(seconds=1) if inclusive_end else _add_months(start, 3)
        return _to_iso(start), _to_iso(end_dt)

    if grain == "year":
        start = base.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        end_dt = start.replace(year=start.year + 1) - timedelta(seconds=1) if inclusive_end else start.replace(year=start.year + 1)
        return _to_iso(start), _to_iso(end_dt)

    return val_iso, val_iso


# =============================
# HÀM CHUẨN HÓA DUCKLING (đã fix năm)
# =============================
def normalize_duckling_times(resp: list, inclusive_end: bool = True, tz: timezone = TZ):
    if not resp:
        return {"type": "none"}

    # ✅ Nếu Duckling nhận 2 mốc thời gian → hiểu là khoảng (from–to)
    if len(resp) >= 2:
        try:
            first_item = resp[0]
            last_item = resp[-1]
            first_val = first_item.get("value", {}).get("value")
            last_val = last_item.get("value", {}).get("value")
            first_grain = first_item.get("value", {}).get("grain", "day")
            last_grain = last_item.get("value", {}).get("grain", "day")

            if first_val and last_val:
                start, _ = _expand_grain_interval(first_val, first_grain, inclusive_end=False, tz=tz)
                _, end = _expand_grain_interval(last_val, last_grain, inclusive_end=True, tz=tz)

                now = datetime.now(TZ)
                if not any(re.search(r"\b20\d{2}\b", x.get("body", "")) for x in resp):
                    s_dt, e_dt = to_vn_timezone(start), to_vn_timezone(end)
                    s_dt = s_dt.replace(year=now.year)
                    e_dt = e_dt.replace(year=now.year)
                    start, end = s_dt.isoformat(), e_dt.isoformat()

                return {"type": "range", "start": start, "end": end, "grain": last_grain}
        except Exception as e:
            print("⚠️ Multi-time normalize error:", e)

    # ✅ Nếu chỉ có 1 mốc thời gian
    item = next((x for x in resp if x.get("dim") == "time"), resp[0])
    primary = None
    top_values = item.get("values")
    top_value = item.get("value")

    if isinstance(top_values, list) and top_values:
        primary = top_values[0]
    elif isinstance(top_value, dict):
        primary = top_value.get("values", [{}])[0] if isinstance(top_value.get("values"), list) else top_value

    if not isinstance(primary, dict):
        return {"type": "none"}

    typ = primary.get("type")
    grain = primary.get("grain")
    body_text = item.get("body", "").lower()

    # ✅ Xử lý kiểu interval
    if typ == "interval":
        start_iso = primary.get("from", {}).get("value")
        end_iso = primary.get("to", {}).get("value")

        if start_iso:
            start_iso = to_vn_timezone(start_iso).isoformat()
        if end_iso:
            end_iso = to_vn_timezone(end_iso).isoformat()

        if inclusive_end and end_iso:
            end_grain = primary.get("to", {}).get("grain") or grain or "day"
            _s, end_iso = _expand_grain_interval(end_iso, end_grain, inclusive_end=True, tz=tz)

        if not re.search(r"\b20\d{2}\b", body_text):
            now = datetime.now(TZ)
            s_dt = to_vn_timezone(start_iso)
            e_dt = to_vn_timezone(end_iso)
            s_dt = s_dt.replace(year=now.year)
            e_dt = e_dt.replace(year=now.year)
            start_iso, end_iso = s_dt.isoformat(), e_dt.isoformat()

        return {"type": "range", "start": start_iso, "end": end_iso}

    # ✅ Xử lý kiểu value (ngày đơn, “hôm nay”)
    if typ == "value":
        val_iso = primary.get("value")
        vn_dt = to_vn_timezone(val_iso)
        now = datetime.now(TZ)

        try:
            if not re.search(r'\b20\d{2}\b', body_text):
                vn_dt = vn_dt.replace(year=now.year)
                # ⚡ Không trừ năm nếu là “hôm nay”, “nay”, “hiện tại”, “bây giờ”
                if vn_dt > now and not re.search(r"hôm nay|nay|hiện tại|bây giờ", body_text, re.IGNORECASE):
                    vn_dt = vn_dt.replace(year=now.year - 1)
        except Exception as e:
            print("⚠️ Year defaulting error:", e)

        val_iso = vn_dt.isoformat()

        if grain in ("week", "month", "quarter", "year"):
            start, end = _expand_grain_interval(val_iso, grain, inclusive_end=inclusive_end, tz=tz)
            return {"type": "range", "start": start, "end": end, "grain": grain}

        start, end = _expand_grain_interval(val_iso, "day", inclusive_end=inclusive_end, tz=tz)
        return {"type": "single", "date": start, "grain": "day"}

    return {"type": "none"}


# =============================
# Intent handling (giữ nguyên)
# =============================
def build_response_with_time(text: str):
    intent, confidence = predict_intent(text)

    time_info = {"type": "none"}

    # ✅ Gọi Duckling cho các intent có thể chứa ngày/tháng
    if any(k in intent for k in ["NGAY", "PAYROLL", "ATTENDANCE", "LUONG", "CONG"]):
        duck_resp = duckling_parse_time(text)
        print(duck_resp)
        time_info = normalize_duckling_times(duck_resp)

    action_text = get_action(intent, text)
    return {"intent": intent, "confidence": confidence, "time": time_info, "message": action_text}


def predict_intent(text):
    try:
        predictions = model.predict(text, k=1)
        intent = predictions[0][0].replace('__label__', '')
        confidence = predictions[1][0]
        return intent, confidence
    except Exception:
        return "UNKNOWN", 0.0


# =============================
# Nội dung phản hồi người dùng
# =============================
def get_action(intent: str, text: str = ""):
    """
    Lấy dữ liệu phản hồi từ listkey (đã lấy từ DB)
    intent = DATA_KEY
    pattern = DATA_NAME
    response = DATA_GROUP
    """
    try:
        for item in listkey:    # listkey = list of dict
            if item["DATA_KEY"] == intent:
                data_name = item.get("DATA_NAME", "")
                data_group = item.get("DATA_GROUP", "")
                # Trả về nội dung bạn muốn (ở đây bạn nói response = DATA_GROUP)
                return f"{data_name} → nhóm dữ liệu: {data_group}"
    except Exception as e:
        print("⚠️ Lỗi khi xử lý get_action:", e)

    return "Xin lỗi, tôi chưa hiểu yêu cầu của bạn. Hãy thử lại nhé! 😊"


# Demo
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"   # hoặc domain cụ thể
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    return response

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    
    res = build_response_with_time(text)
    
    return jsonify(res)

@app.route('/pushKey', methods=['POST'])
def push_key():
    try:
        data = request.get_json()  # nhận JSON từ frontend
        key = data.get('key')
        pattern = data.get('pattern')

        if not key or not pattern:
            return jsonify({"error": "Thiếu key hoặc pattern"}), 400

        # Đổ dữ liệu vào 2 mảng Python
        keys_list.append(key)
        patterns_list.append(pattern)

        return jsonify({
            "message": "Đã nhận dữ liệu thành công!",
            "keys": keys_list,
            "patterns": patterns_list
        }), 200

    except Exception as e:
        print("❌ Lỗi:", e)
        return jsonify({"error": str(e)}), 500



@app.route('/getKeys', methods=['GET'])
def get_keys():
    """
    Trả về danh sách keys và patterns hiện có.
    """
    try:
        return jsonify({
            "keys": keys_list,
            "patterns": patterns_list
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)