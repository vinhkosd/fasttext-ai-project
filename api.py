import fasttext  # type: ignore
import re
import requests     # type: ignore
from datetime import datetime, timezone, timedelta
from typing import Optional
from flask import Flask, request, jsonify
from dateutil import parser  # type: ignore
from dateutil import tz  # type: ignore

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

    # Bổ sung: nếu có dạng "1/10" hoặc "12/9" nhưng KHÔNG có chữ 'tháng' hay 'năm' => thêm vào
    def repl_short_date(m):
        d, mth = int(m.group(1)), int(m.group(2))
        return f"ngày {d} tháng {mth}"

    if not re.search(r"\btháng\b", text):
        text = re.sub(r"\b(\d{1,2})/(\d{1,2})\b", repl_short_date, text)

    # Ép kiểu dd/mm/yyyy thành "ngày dd tháng mm năm yyyy"
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
    original_text = text.strip().lower()

    if re.search(r"\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b", original_text):
        if not original_text.startswith("ngày"):
            original_text = "ngày " + original_text

    if re.search(r"\b\d{1,2}/\d{4}\b", original_text) and "tháng" not in original_text:
        original_text = "tháng " + original_text

    if re.fullmatch(r".*\b\d{4}\b.*", original_text) and "/" not in original_text and "năm" not in original_text:
        original_text = "năm " + original_text

    text = preprocess_date_text(original_text)
    if ref_time is None:
        ref_time = datetime.now(TZ)

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

def to_vn_timezone(dt_str: str):
    try:
        dt = parser.isoparse(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        vn_tz = tz.gettz("Asia/Ho_Chi_Minh")
        return dt.astimezone(vn_tz)
    except Exception as e:
        print(f"⚠️ to_vn_timezone error: {e}")
        return parser.isoparse(dt_str)

def _expand_grain_interval(val_iso: str, grain: str, inclusive_end: bool = True, tz: timezone = TZ):
    base = to_vn_timezone(val_iso)

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
# HÀM CHUẨN HÓA DUCKLING (fix năm)
# =============================
def normalize_duckling_times(resp: list, original_text: str = "", inclusive_end: bool = True, tz: timezone = TZ):
    """
    Chuẩn hóa kết quả Duckling về dạng consistent:
    - Xử lý các từ khóa "hôm nay/ngày mai/hôm qua"
    - Xử lý khoảng thời gian thủ công (bao gồm "ngày dd/mm/yyyy đến ngày dd/mm/yyyy")
    - Xử lý ngày riêng lẻ (trả start/end giống nhau)
    - Xử lý multi-time (nhiều item) + cross-year thông minh
    - Fallback Duckling interval/value
    """
    text = (original_text or "").strip().lower()
    now = datetime.now(TZ)
    today = datetime(now.year, now.month, now.day, tzinfo=TZ)

    # -----------------------
    # Các từ khóa rõ ràng
    # -----------------------
    if re.search(r"\bhôm nay\b", text):
        return {"type": "single", "grain": "day", "date_start": today.isoformat(), "date_end": today.isoformat()}

    if re.search(r"\bngày mai\b", text):
        t = today + timedelta(days=1)
        return {"type": "single", "grain": "day", "date_start": t.isoformat(), "date_end": t.isoformat()}

    if re.search(r"\bhôm qua\b", text):
        t = today - timedelta(days=1)
        return {"type": "single", "grain": "day", "date_start": t.isoformat(), "date_end": t.isoformat()}

    # -----------------------
    # Khoảng thời gian thủ công (có thể có từ "ngày " trước số)
    # -----------------------
    m = re.search(
        r"(?:ngày\s*)?(\d{1,2}[/-]\d{1,2}(?:[/-]\d{4})?)\s*(?:đến|tới|-|->)\s*(?:ngày\s*)?(\d{1,2}[/-]\d{1,2}(?:[/-]\d{4})?)",
        text
    )
    if m:
        start_str, end_str = m.groups()

        def parse_date(dstr: str):
            parts = list(map(int, re.split(r"[/-]", dstr)))
            if len(parts) == 3:
                d, mth, y = parts
            elif len(parts) == 2:
                d, mth = parts
                y = now.year
            else:
                return None
            return datetime(y, mth, d, tzinfo=TZ)

        start_dt = parse_date(start_str)
        end_dt = parse_date(end_str)

        if start_dt and end_dt:
            if end_dt < start_dt:
                start_dt, end_dt = end_dt, start_dt
            end_dt = end_dt.replace(hour=23, minute=59, second=59)
            return {"type": "range", "grain": "day", "start": start_dt.isoformat(), "end": end_dt.isoformat()}

    # -----------------------
    # Ngày riêng lẻ dd/mm[/yyyy]
    # -----------------------
    m = re.search(r"(\d{1,2})[/-](\d{1,2})(?:[/-](\d{4}))?", text)
    if m:
        d, mth, y = m.groups()
        d, mth = int(d), int(mth)
        y = int(y) if y else now.year
        dt = datetime(y, mth, d, tzinfo=TZ)
        return {"type": "single", "grain": "day", "date_start": dt.isoformat(), "date_end": dt.isoformat()}

    # -----------------------
    # Multi-time / nhiều item Duckling
    # -----------------------
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

                s_dt, e_dt = to_vn_timezone(start), to_vn_timezone(end)
                now_year = now.year

                # Nếu không có năm trong text, đặt năm cho start/end
                s_dt = s_dt.replace(year=now_year)

                # Nếu end month < start month → cross-year
                if e_dt.month < s_dt.month:
                    e_dt = e_dt.replace(year=now_year + 1)
                else:
                    e_dt = e_dt.replace(year=now_year)

                start, end = s_dt.isoformat(), e_dt.isoformat()
                return {"type": "range", "start": start, "end": end, "grain": last_grain}
        except Exception as e:
            print("⚠️ Multi-time normalize error:", e)

    # -----------------------
    # Fallback Duckling
    # -----------------------
    if not resp:
        return {"type": "none"}

    item = next((x for x in resp if x.get("dim") == "time"), resp[0])
    primary = item.get("value", {})
    typ = primary.get("type")
    grain = primary.get("grain", "day")

    if typ == "interval":
        start_iso = primary.get("from", {}).get("value")
        end_iso = primary.get("to", {}).get("value")
        if start_iso:
            start_iso = to_vn_timezone(start_iso).isoformat()
        if end_iso:
            end_iso = to_vn_timezone(end_iso).isoformat()
        return {"type": "range", "grain": grain, "start": start_iso, "end": end_iso}

    if typ == "value":
        val_iso = primary.get("value")
        if grain in ("week", "month", "quarter", "year"):
            start, end = _expand_grain_interval(val_iso, grain, inclusive_end=inclusive_end, tz=tz)
            return {"type": "range", "grain": grain, "start": start, "end": end}
        start, _ = _expand_grain_interval(val_iso, "day", inclusive_end=inclusive_end, tz=tz)
        return {"type": "single", "grain": "day", "date_start": start, "date_end": start}

    return {"type": "none"}



# =============================
# Intent handling (giữ nguyên)
# =============================
def build_response_with_time(text: str):
    intent, confidence = predict_intent(text)

    if "WEL" not in intent and confidence < 0.7:
        intent = "FALLBACK"

    time_info = {"type": "none"}
    if any(k in intent for k in ["NGAY", "PAYROLL", "ATTENDANCE", "LUONG", "CONG"]):
        duck_resp = duckling_parse_time(text)
        time_info = normalize_duckling_times(duck_resp, original_text=text)

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
def get_action(intent, text):
    if intent == "WELCOME":
        return "Xin chào 👋 Tôi có thể giúp gì cho bạn?"
    elif intent == "PAYROLL_PERSONAL":
        return "Để tôi kiểm tra thông tin lương cho bạn nhé."
    elif intent == "ATTENDANCE_PERSONAL":
        return "Hãy để tôi xem ngày công của bạn."
    elif intent == "FALLBACK":
        return "Xin lỗi, tôi chưa hiểu câu hỏi của bạn. Bạn có thể nói lại không?"
    else:
        return "Xin lỗi, tôi chưa có thông tin cho yêu cầu này."


# =============================
# Flask routes
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

    res = build_response_with_time(text)
    return jsonify(res)

@app.route('/pushKey', methods=['POST'])
def push_key():
    try:
        data = request.get_json()
        key = data.get('key')
        pattern = data.get('pattern')

        if not key or not pattern:
            return jsonify({"error": "Thiếu key hoặc pattern"}), 400

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
    try:
        return jsonify({
            "keys": keys_list,
            "patterns": patterns_list
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
