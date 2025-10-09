from datetime import datetime, timedelta, timezone

TZ = timezone(timedelta(hours=7))  # Asia/Ho_Chi_Minh

def get_quick_replies(intent: str, time_info: dict):
    quick_replies = []

    # Nếu là fallback → gợi ý lại các chức năng chính
    if intent == "FALLBACK":
        return [
            {"label": "💰 Xem lương", "payload": "xem lương tháng này"},
            {"label": "📅 Xem ngày công", "payload": "xem ngày công hôm nay"},
            {"label": "🕒 Xem nghỉ phép", "payload": "xem nghỉ phép tháng này"},
        ]

    # Nếu không có time rõ ràng
    if not time_info or time_info.get("type") == "none":
        return [
            {"label": "💰 Xem lương tháng này", "payload": "xem lương tháng này"},
            {"label": "📅 Xem chấm công hôm nay", "payload": "xem chấm công hôm nay"},
            {"label": "🕒 Xem nghỉ phép tháng này", "payload": "xem nghỉ phép tháng này"},
        ]

    grain = time_info.get("grain")
    now = datetime.now(TZ)

    # --- Nếu là ngày ---
    if grain == "day":
        return [
            {"label": "🔙 Hôm qua", "payload": "xem lương ngày hôm qua"},
            {"label": "📅 Hôm nay", "payload": "xem lương ngày hôm nay"},
            {"label": "⏭️ Ngày mai", "payload": "xem lương ngày mai"},
            {"label": "💰 Xem tháng này", "payload": "xem lương tháng này"},
        ]

    # --- Nếu là tháng ---
    if grain == "month":
        return [
            {"label": "⬅️ Tháng trước", "payload": "xem lương tháng trước"},
            {"label": "📅 Tháng này", "payload": "xem lương tháng này"},
            {"label": "➡️ Tháng sau", "payload": "xem lương tháng sau"},
            {"label": "📆 Năm nay", "payload": "xem lương năm nay"},
        ]

    # --- Nếu là năm ---
    if grain == "year":
        return [
            {"label": "⬅️ Năm trước", "payload": "xem lương năm trước"},
            {"label": "📅 Năm nay", "payload": "xem lương năm nay"},
            {"label": "➡️ Năm sau", "payload": "xem lương năm sau"},
        ]

    # --- Mặc định fallback ---
    return [
        {"label": "💰 Xem lương tháng này", "payload": "xem lương tháng này"},
        {"label": "📅 Xem chấm công hôm nay", "payload": "xem chấm công hôm nay"},
        {"label": "🕒 Xem nghỉ phép tháng này", "payload": "xem nghỉ phép tháng này"},
    ]
