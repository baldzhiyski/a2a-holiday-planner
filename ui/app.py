import json
import time
import requests
import streamlit as st

st.set_page_config(page_title="A2A Trip Planner", layout="wide")

# --- Config ---
HOST_URL = st.secrets.get("HOST_URL", "http://localhost:12020")
SEND_URL = f"{HOST_URL}/messages:sendMessage"

st.title("🧭 A2A Trip Planner — Dashboard")

with st.sidebar:
    st.subheader("Host Endpoint")
    host_url = st.text_input("Host base URL", HOST_URL)
    send_url = f"{host_url}/messages:sendMessage"
    st.caption("Make sure your Host is running (uv run -m host.main).")

# --- Prompt form ---
with st.form("prompt_form"):
    st.subheader("Request")
    user_prompt = st.text_area(
        "Your instruction",
        value="Plan a 4-day trip to Lisbon from Berlin, 2025-11-10 to 2025-11-14, for 2 people. "
              "Budget €2200 total. Prefer walkable areas, boutique hotels (>=4★). "
              "Depart after 09:00, return after 14:00. Avoid redeyes. Book the best option.",
        height=150,
    )
    context_id = st.text_input("Context ID", value="ctx1")
    submitted = st.form_submit_button("Send to Host")

def _make_send_message(text: str, ctx: str):
    # Build A2A payload; id/messageId/taskId can be same for demo
    msg_id = f"msg-{int(time.time())}"
    payload = {
        "id": msg_id,
        "params": {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": text}],
                "messageId": msg_id,
                "taskId": f"task-{msg_id}",
                "contextId": ctx,
            }
        }
    }
    return payload

def _post_json(url: str, payload: dict):
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()

def _collect_artifacts(a2a_response: dict):
    """Extract text parts from artifacts in the A2A Task response."""
    try:
        root = a2a_response.get("root") or a2a_response  # some servers wrap differently
        # work with the canonical dict (DefaultRequestHandler returns {"root":{"result":{"artifacts":[...]}}})
        result = root.get("result", {})
        arts = result.get("artifacts", [])
        texts = []
        for art in arts:
            for p in art.get("parts", []):
                if p.get("type") == "text":
                    texts.append(p.get("text", ""))
        return texts, arts
    except Exception:
        return [], []

def _best_json(texts):
    """Pick the largest valid JSON blob among text parts."""
    best = None
    best_len = -1
    for t in texts:
        try:
            obj = json.loads(t)
            size = len(json.dumps(obj))
            if size > best_len:
                best = obj
                best_len = size
        except Exception:
            continue
    return best

# --- Send prompt ---
if submitted:
    with st.spinner("Contacting Host…"):
        payload = _make_send_message(user_prompt, context_id)
        a2a_resp = _post_json(send_url, payload)

    st.success("Host responded")
    with st.expander("Raw A2A response", expanded=False):
        st.code(json.dumps(a2a_resp, indent=2), language="json")

    texts, arts = _collect_artifacts(a2a_resp)
    structured = _best_json(texts)  # likely the candidates array or booking confirmation

    # Try to show candidates if present
    candidates = None
    confirmation = None

    if isinstance(structured, dict) and "booking" in structured:
        confirmation = structured
    elif isinstance(structured, dict) and "candidates" in structured:
        candidates = structured["candidates"]
    elif isinstance(structured, list):
        candidates = structured

    col_left, col_right = st.columns([2,1])

    with col_left:
        st.subheader("Result")
        # 1) Booking confirmation
        if confirmation:
            b = confirmation["booking"]
            st.success("✅ Booking confirmed")
            st.write(f"**Flights:** {b['flights_confirmation']}")
            st.write(f"**Hotel:** {b['hotel_confirmation']}")
            st.write(f"**Activities booked:** {b['activities_count']}")
            st.divider()

        # 2) Candidate itineraries
        if candidates:
            st.info("Top itinerary candidates")
            for idx, c in enumerate(candidates, start=1):
                with st.container(border=True):
                    st.markdown(f"### Option {idx}: {c['summary']}")
                    pb = c.get("price_breakdown_eur", {})
                    total = c.get("total_eur")
                    st.write(f"**Total:** €{total:.0f}  "
                             f"(Flights €{pb.get('outbound',0)+pb.get('inbound',0):.0f} • "
                             f"Hotel €{pb.get('hotel',0):.0f} • "
                             f"Activities €{pb.get('activities',0):.0f})")

                    o = c["outbound"]; i = c["inbound"]; h = c["hotel"]
                    st.write(f"**Outbound:** {o['source']} → {o['dest']} · {o['depart_iso']} → {o['arrive_iso']} · {o['airline']}")
                    st.write(f"**Inbound:** {i['source']} → {i['dest']} · {i['depart_iso']} → {i['arrive_iso']} · {i['airline']}")
                    st.write(f"**Hotel:** {h['name']} (⭐ {h['rating']}) · {h['checkin_iso']} → {h['checkout_iso']}")

                    with st.expander("Show daily plan"):
                        for day in c["days"]:
                            st.write(f"- **{day['date_iso']}** — "
                                     f"morning: {day.get('morning') or '—'}, "
                                     f"afternoon: {day.get('afternoon') or '—'}, "
                                     f"evening: {day.get('evening') or '—'}")

                    # Book button posts a “Book option N”
                    if st.button(f"Book option {idx}", key=f"book_{idx}"):
                        book_payload = _make_send_message(f"Book option {idx}", context_id)
                        book_resp = _post_json(send_url, book_payload)
                        bt, _ = _collect_artifacts(book_resp)
                        conf = _best_json(bt)
                        if conf and isinstance(conf, dict) and conf.get("status") == "success":
                            st.success("Booking confirmed!")
                            st.json(conf)
                        else:
                            st.error("Booking failed or no confirmation returned.")
        else:
            # fallback: show plain text parts
            st.write("No candidates parsed; showing returned texts:")
            for t in texts:
                st.code(t)

    with col_right:
        st.subheader("Artifacts")
        if arts:
            for art in arts:
                st.caption(f"Artifact: {art.get('name','(unnamed)')}")
                for p in art.get("parts", []):
                    if p.get("type") == "text":
                        st.code(p["text"][:5000], language="json")
                st.divider()
        else:
            st.write("No artifacts detected.")
