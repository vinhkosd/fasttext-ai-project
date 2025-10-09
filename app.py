import fasttext  # type: ignore
import re
import requests  # type: ignore
from datetime import datetime, timezone, timedelta
from typing import Optional
from flask import Flask, request, jsonify
from dateutil import parser, tz  # type: ignore
from quick_replies import get_quick_replies

# =========================================================
# ⚙️ Cấu hình ban đầu
# =========================================================
model = fasttext.load_model("models/intent_model.bin")
DUCKLING_URL = "http://localhost:8085/parse"
VI_LOCALE = "vi_VN"
TZ = timezone(timedelta(hours=7))  # Asia/Ho_Chi_Minh

app = Flask(__name__)

# =========================================================
# ✅ Cho phép CORS để frontend gọi được
# =========================================================
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    return response

# =========================================================
# ✅ Các hàm xử lý thời gian & intent
# =========================================================
def preprocess_date_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\bthang\b", "tháng", text)
    text = re.sub(r"\bnam\b", "năm", text)
    text = re.sub(r"[-\.]", "/", text)
    text = re.sub(r"\s*/\s*", "/", text)
    def repl_short_date(m):
        d, mth = int(m.group(1)), int(m.group(2))
        return f"ngày {d} tháng {mth}"
    if not re.search(r"\btháng\b", text):
        text = re.sub(r"\b(\d{1,2})/(\d{1,2})\b", repl_short_date, text)
    text = re.sub(
        r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b",
        lambda m: f"ngày {int(m.group(1))} tháng {int(m.group(2))} năm {m.group(3)}",
        text,
    )
    return text


def duckling_parse_time(text: str, ref_time: Optional[datetime] = None):
    text = preprocess_date_text(text)
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
            timeout=5,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("⚠️ Duckling error:", e)
        return []


def to_vn_timezone(dt_str: str):
    try:
        dt = parser.isoparse(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        vn_tz = tz.gettz("Asia/Ho_Chi_Minh")
        return dt.astimezone(vn_tz)
    except Exception:
        return parser.isoparse(dt_str)


def _end_of_month(dt: datetime) -> datetime:
    next_month = dt.replace(day=28) + timedelta(days=4)
    return next_month - timedelta(days=next_month.day)


def normalize_duckling_times(resp: list, original_text: str = ""):
    now = datetime.now(TZ)
    if not resp:
        return {"type": "none"}

    item = resp[0]
    primary = item.get("value", {})
    grain = primary.get("grain", "day")
    val_iso = primary.get("value")
    vn_dt = to_vn_timezone(val_iso)

    if grain == "day":
        d = vn_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return {"type": "single", "grain": "day", "date": d.isoformat()}

    if grain == "month":
        start = vn_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = _end_of_month(start)
        return {
            "type": "range",
            "grain": "month",
            "start": start.isoformat(),
            "end": end.isoformat(),
        }

    if grain == "year":
        start = datetime(vn_dt.year, 1, 1, tzinfo=TZ)
        end = datetime(vn_dt.year, 12, 31, 23, 59, 59, tzinfo=TZ)
        return {
            "type": "range",
            "grain": "year",
            "start": start.isoformat(),
            "end": end.isoformat(),
        }

    return {"type": "none"}


def predict_intent(text: str):
    try:
        predictions = model.predict(text, k=1)
        intent = predictions[0][0].replace("__label__", "")
        confidence = predictions[1][0]
        return intent, confidence
    except Exception:
        return "UNKNOWN", 0.0


def get_action(intent, text):
    if intent == "WELCOME":
        return (
            "Chào bạn nha! 👋 \n"
            "Tôi có thể giúp bạn các việc sau:\n"
            "📊 **Xem lương** - Xem bảng lương cá nhân\n"
            "📅 **Xem chấm công** - Xem thông tin chấm công\n"
            "👤 **Xem thông tin cá nhân** - Xem hồ sơ cá nhân\n"
            "📋 **Xem ngày nghỉ** - Xem thông tin nghỉ phép\n\n"
            "💡 **Ví dụ cách hỏi:**\n"
            "- \"Cho tôi xem lương tháng này\"\n"
            "- \"Xem chấm công từ 1/10 đến 31/10\"\n"
            "- \"Hiển thị thông tin cá nhân\"\n"
            "- \"Chấm công tháng trước\"\n\n"
            "Hãy cho tôi biết bạn cần gì nhé! 😊"
        )
    elif intent == "PAYROLL_PERSONAL":
        return "Để tôi kiểm tra thông tin lương cho bạn nhé."
    elif intent == "ATTENDANCE_PERSONAL":
        return "Hãy để tôi xem ngày công của bạn."
    elif intent == "FALLBACK":
        return "Xin lỗi, tôi chưa hiểu câu hỏi của bạn. Bạn có thể nói lại không?"
    else:
        return "Xin lỗi, tôi chưa có thông tin cho yêu cầu này."


def build_response_with_time(text: str):
    intent, confidence = predict_intent(text)
    if "WEL" not in intent and confidence < 0.7:
        intent = "FALLBACK"

    time_info = {"type": "none"}
    if any(k in intent for k in ["NGAY", "PAYROLL", "ATTENDANCE", "LUONG", "CONG"]):
        duck_resp = duckling_parse_time(text)
        time_info = normalize_duckling_times(duck_resp, text)

    action_text = get_action(intent, text)
    quick_replies = get_quick_replies(intent, time_info)

    return {
        "intent": intent,
        "confidence": round(confidence, 2),
        "time": time_info,
        "message": action_text,
        "quick_replies": quick_replies,
        "response_type": "text",
    }

# =========================================================
# ✅ Flask route /predict
# =========================================================
@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json(force=True)
    text = data.get("text", "")
    result = build_response_with_time(text)
    return jsonify(result)


# =========================================================
# ✅ Main entry
# =========================================================
if __name__ == "__main__":
    app.run(debug=True, port=5000)
