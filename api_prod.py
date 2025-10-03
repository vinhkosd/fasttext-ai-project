import fasttext
import re
import requests
from datetime import datetime, timezone, timedelta
from typing import Optional
from flask import Flask, request, jsonify

app = Flask(__name__)
# Load model
model = fasttext.load_model('models/intent_model.bin')
DUCKLING_URL = "http://localhost:8085/parse"
VI_LOCALE = "vi_VN"
TZ = timezone(timedelta(hours=7))  # Asia/Ho_Chi_Minh

def duckling_parse_time(text: str, ref_time: Optional[datetime] = None):
    """
    Gọi Duckling server để parse ngày/giờ.
    """
    if ref_time is None:
        ref_time = datetime.now(TZ)
    print("Duckling đang xử lí")
    reftime_ms = int(ref_time.timestamp() * 1000)
    # Duckling yêu cầu body x-www-form-urlencoded, không phải JSON
    data = {
        "locale": VI_LOCALE,
        "text": text,
        "dims": '["time"]',
        "reftime": str(reftime_ms),
    }
    try:
        r = requests.post(
            DUCKLING_URL,
            data=data,  # form-urlencoded
            headers={"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"},
            timeout=5
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("Duckling error:", e)
        return []

def _iso_to_dt(s: str) -> datetime:
    # Hỗ trợ cả 'Z'
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

def _expand_grain_interval(val_iso: str, grain: str, inclusive_end: bool = True, tz: timezone = TZ):
    base = _iso_to_dt(val_iso).astimezone(tz)

    if grain == "day":
        start = base.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        if inclusive_end:
            end = end - timedelta(seconds=1)
        return _to_iso(start), _to_iso(end)

    if grain == "week":
        # Duckling thường trả đầu tuần; ta chuẩn hoá: start = ngày đó 00:00
        start = base.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=7)
        if inclusive_end:
            end = end - timedelta(seconds=1)
        return _to_iso(start), _to_iso(end)

    if grain == "month":
        start = base.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if inclusive_end:
            end_dt = _end_of_month(start)
        else:
            end_dt = _add_months(start, 1)  # exclusive
        return _to_iso(start), _to_iso(end_dt)

    if grain == "quarter":
        # Tính quý: 1–3, 4–6, 7–9, 10–12
        q = (base.month - 1) // 3
        start_month = q * 3 + 1
        start = base.replace(month=start_month, day=1, hour=0, minute=0, second=0, microsecond=0)
        if inclusive_end:
            end_dt = _add_months(start, 3) - timedelta(seconds=1)
        else:
            end_dt = _add_months(start, 3)
        return _to_iso(start), _to_iso(end_dt)

    if grain == "year":
        start = base.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        if inclusive_end:
            end_dt = start.replace(year=start.year + 1) - timedelta(seconds=1)
        else:
            end_dt = start.replace(year=start.year + 1)
        return _to_iso(start), _to_iso(end_dt)

    # Mặc định: coi như single
    return val_iso, val_iso

def normalize_duckling_times(resp: list, inclusive_end: bool = True, tz: timezone = TZ):
    """
    Hỗ trợ cả 2 dạng:
    - item["values"] (list candidates)
    - item["value"] (single object), có thể chứa 'values' bên trong.
    """
    if not resp:
        return {"type": "none"}

    # Ưu tiên dim='time'
    item = next((x for x in resp if x.get("dim") == "time"), resp[0])

    # Lấy primary candidate
    primary = None
    top_values = item.get("values")
    top_value = item.get("value")

    if isinstance(top_values, list) and top_values:
        primary = top_values[0]
    elif isinstance(top_value, dict):
        # Một số bản trả 'value' là object duy nhất; đôi khi value còn chứa 'values'
        if isinstance(top_value.get("values"), list) and top_value["values"]:
            primary = top_value["values"][0]
        else:
            primary = top_value

    if not isinstance(primary, dict):
        return {"type": "none"}

    typ = primary.get("type")
    grain = primary.get("grain")

    # Interval (từ...đến...)
    if typ == "interval":
        start_iso = primary.get("from", {}).get("value")
        end_iso = primary.get("to", {}).get("value")
        # Nếu muốn inclusive end theo grain (nếu Duckling trả grain ở from/to)
        if inclusive_end and end_iso:
            # Cố gắng dùng grain nếu có ở 'to', nếu không dùng 'day'
            end_grain = primary.get("to", {}).get("grain") or grain or "day"
            if end_grain in ("day", "week", "month", "quarter", "year"):
                _s, end_iso = _expand_grain_interval(end_iso, end_grain, inclusive_end=True, tz=tz)
        return {"type": "range", "start": start_iso, "end": end_iso}

    # Value (mốc đơn). Với grain rộng → đổi thành range
    if typ == "value":
        val_iso = primary.get("value")
        if grain in ("week", "month", "quarter", "year"):
            start, end = _expand_grain_interval(val_iso, grain, inclusive_end=inclusive_end, tz=tz)
            return {"type": "range", "start": start, "end": end, "grain": grain}
        return {"type": "single", "date": val_iso, "grain": grain}

    return {"type": "none"}

def build_response_with_time(text: str):
    intent, confidence = predict_intent(text)

    time_info = {"type": "none"}
    # Nếu intent liên quan thời gian thì gọi Duckling
    if "NGAY" in intent:
        duck_resp = duckling_parse_time(text)
        print(duck_resp)
        time_info = normalize_duckling_times(duck_resp)

    action_text = get_action(intent, text)
    return {
        "intent": intent,
        "confidence": confidence,
        "time": time_info,
        "message": action_text,
    }

def predict_intent(text):
    """Dự đoán intent với xử lý lỗi"""
    try:
        predictions = model.predict(text, k=1)
        intent = predictions[0][0].replace('__label__', '')
        confidence = predictions[1][0]
        return intent, confidence
    except Exception as e:
        return "UNKNOWN", 0.0

def get_action(intent, text=""):
    actions = {
        "WELCOME": """Chào bạn nha! 👋 
        Tôi có thể giúp bạn các việc sau:
        📊 **Xem lương** - Xem bảng lương cá nhân
        📅 **Xem chấm công** - Xem thông tin chấm công
        👤 **Xem thông tin cá nhân** - Xem hồ sơ cá nhân
        📋 **Xem ngày nghỉ** - Xem thông tin nghỉ phép

        💡 **Ví dụ cách hỏi:**
        - "Cho tôi xem lương tháng này"
        - "Xem chấm công từ 1/10 đến 31/10"  
        - "Hiển thị thông tin cá nhân"
        - "Chấm công tháng trước"

        Hãy cho tôi biết bạn cần gì nhé! 😊""",
        
        "HELP_INFORMATION": """Xin chào, tôi là TimeAI! 🤖
        Tôi có thể giúp bạn những thông tin:
        • 📋 **Thông tin cá nhân** - Họ tên, mã NV, phòng ban, chức vụ
        • 📅 **Thông tin ngày công** - Chấm công, giờ làm, tăng ca  
        • 🏖️ **Thông tin ngày nghỉ** - Phép năm, ngày vắng
        • 💰 **Thông tin lương tháng** - Bảng lương, thu nhập

        Bạn muốn xem thông tin nào?""",
        
        "HELP_PERSONAL": """Tôi có thể hỗ trợ thông tin liên quan đến thông tin cá nhân của bạn: 
        • 👤 Họ tên
        • 🔢 Mã nhân viên  
        • 🏢 Phòng ban
        • 💼 Chức vụ
        • 📝 Công việc

        Bạn muốn xem thông tin cụ thể nào?""",
        
        "NGAYCONG_MON": """Vâng, đây là dữ liệu chấm công của bạn từ đầu tháng đến hôm nay:

    📊 **Bảng chấm công tháng 10/2025**
        Ngày làm việc Ca làm việc Giờ vào Giờ ra Giờ làm Giờ tăng ca Loại vắng Số giờ vắng
        05/10/2025 08:00-17:00 08:00 17:40 8 0 - -
        06/10/2025 08:00-17:00 07:55 18:30 8 1 - -
        07/10/2025 08:00-17:00 - - - - Phép năm 8""",
    "NGAYCONG_TODAY": f"""Vâng, đây là dữ liệu chấm công của bạn ngày hôm nay:

        📅 **Ngày làm việc**: {datetime.now().strftime('%d/%m/%Y')} 
        ⏰ **Ca làm việc**: 08:00 - 17:00 (nghỉ trưa 12:00-13:00)
        🟢 **Giờ vào**: 08:10  
        🔴 **Giờ ra**: Chưa có
        💡 **Trạng thái**: Đang làm việc""",
                
        "NGAYCONG_YESTERDAY": """Vâng, đây là dữ liệu chấm công của bạn ngày hôm qua:

        📅 **Ngày làm việc**: 19/10/2025 (Thứ 4)
        ⏰ **Ca làm việc**: 08:00 - 17:00 (nghỉ trưa 12:00-13:00)
        🟢 **Giờ vào**: 08:15 (Trễ 15 phút)
        🔴 **Giờ ra**: 17:10
        ⏱️ **Giờ làm việc**: 7.5
        🌙 **Giờ tăng ca thực tế**: 2
        ✅ **Giờ tăng ca được duyệt**: 2
        ❌ **Giờ vắng**: Không có
        📋 **Loại vắng**: Không có""",
        
        "NGAYCONG_FROMTO": """Vâng, đây là dữ liệu chấm công của bạn từ ngày 05/10/2025 đến 30/10/2025:

    📊 **Bảng chấm công**
        Ngày làm việc Ca làm việc Giờ vào Giờ ra Giờ làm Giờ tăng ca Loại vắng Số giờ vắng
        05/10/2025 08:00-17:00 08:00 17:40 8 0 - -
        06/10/2025 08:00-17:00 07:55 18:30 8 1 - -
        07/10/2025 08:00-17:00 - - - - Phép năm 8
        ... (các ngày khác)

""","NGAYPHEPNAM_YEAR": """Vâng, đây là dữ liệu chi tiết về ngày nghỉ phép năm của bạn:

    📋 **Phép năm đã sử dụng:**
        • 📅 05/01/2025 : 8 giờ
        • 📅 12/02/2025 : 4 giờ  
        • 📅 25/04/2025 : 8 giờ

    📊 **Tổng kết:**
        • ✅ Tổng đã nghỉ phép năm: 20 giờ
        • 🎯 Phép năm còn lại: 2 ngày (16 giờ)""",
        
        "NGAYPHEPNAM_FROMTO": """Vâng, đây là dữ liệu chi tiết về ngày nghỉ phép năm từ ngày 01/05/2025 đến 30/10/2025 của bạn:

    📋 **Phép năm trong khoảng thời gian:**
        • 📅 05/01/2025 : 8 giờ
        • 📅 12/02/2025 : 4 giờ
        • 📅 25/04/2025 : 8 giờ

    📊 **Tổng kết:**
        • ✅ Tổng đã nghỉ phép năm: 20 giờ
        • 🎯 Phép năm còn lại: 2 ngày (16 giờ)""",
        
        "NGAYNGHI_YEAR": """Vâng, đây là dữ liệu ngày nghỉ của bạn trên hệ thống ghi nhận từ đầu năm đến nay:

    📊 **Bảng ngày nghỉ**
        Ngày làm việc Ca làm việc Loại vắng Số giờ vắng
        05/10/2025 08:00-17:00 Phép năm 8
        06/10/2025 08:00-17:00 Không phép 8
        07/10/2025 08:00-17:00 Phép năm 4

"""}
    return actions.get(intent, "Xin lỗi, tôi chưa hiểu yêu cầu của bạn. Hãy thử lại nhé! 😊")

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

if __name__ == '__main__':
    from gevent.pywsgi import WSGIServer
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()