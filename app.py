import os
import cv2
import time
import sqlite3
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import folium
from streamlit_folium import st_folium
from ultralytics import YOLO


# =====================================================
# APP CONFIG
# =====================================================

st.set_page_config(
    page_title="SARathi - Search and Rescue Prototype",
    page_icon="🚁",
    layout="wide"
)

DB_PATH = "sarathi_incidents.db"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

APP_TITLE = "SARathi"
APP_SUBTITLE = "Offline Search-and-Rescue Coordination Prototype"

# COCO class IDs used by YOLOv8
PERSON_CLASS_ID = 0

# Treat vehicles/large movable objects as possible obstruction indicators.
OBSTRUCTION_CLASS_IDS = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    8: "boat"
}


# =====================================================
# DATABASE
# =====================================================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            source TEXT,
            label TEXT,
            confidence REAL,
            latitude REAL,
            longitude REAL,
            status TEXT,
            notes TEXT,
            image_path TEXT
        )
        """
    )

    conn.commit()
    conn.close()


def add_incident(
    source,
    label,
    confidence,
    latitude,
    longitude,
    status="New",
    notes="",
    image_path=""
):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO incidents
        (
            created_at,
            source,
            label,
            confidence,
            latitude,
            longitude,
            status,
            notes,
            image_path
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            source,
            label,
            float(confidence),
            float(latitude),
            float(longitude),
            status,
            notes,
            image_path
        )
    )

    conn.commit()
    conn.close()


def get_incidents():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM incidents ORDER BY id DESC",
        conn
    )
    conn.close()
    return df


def update_incident_status(incident_id, status):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        "UPDATE incidents SET status = ? WHERE id = ?",
        (status, incident_id)
    )

    conn.commit()
    conn.close()


def delete_all_incidents():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM incidents")
    conn.commit()
    conn.close()


# =====================================================
# MODEL
# =====================================================

@st.cache_resource
def load_model():
    """
    YOLOv8n is small and downloads automatically on first run.
    It works on CPU, but GPU will be faster.
    """
    model = YOLO("yolov8n.pt")
    return model


def classify_detection(class_id, confidence):
    if class_id == PERSON_CLASS_ID:
        return "Candidate Person", confidence

    if class_id in OBSTRUCTION_CLASS_IDS:
        return f"Possible Obstruction: {OBSTRUCTION_CLASS_IDS[class_id]}", confidence

    return None, confidence


def detect_objects(image_pil, confidence_threshold):
    model = load_model()

    image_rgb = np.array(image_pil.convert("RGB"))
    results = model.predict(
        source=image_rgb,
        conf=confidence_threshold,
        verbose=False
    )

    detections = []
    annotated = image_rgb.copy()

    if len(results) == 0:
        return detections, Image.fromarray(annotated)

    result = results[0]

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        label, confidence = classify_detection(class_id, confidence)

        if label is None:
            continue

        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy

        if "Person" in label:
            colour = (40, 220, 150)
        else:
            colour = (255, 170, 60)

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            colour,
            3
        )

        caption = f"{label} {confidence:.2f}"

        cv2.rectangle(
            annotated,
            (x1, max(y1 - 32, 0)),
            (x1 + min(420, len(caption) * 13), y1),
            colour,
            -1
        )

        cv2.putText(
            annotated,
            caption,
            (x1 + 5, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

        detections.append(
            {
                "label": label,
                "confidence": confidence,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "center_x": int((x1 + x2) / 2),
                "center_y": int((y1 + y2) / 2)
            }
        )

    return detections, Image.fromarray(annotated)


def save_image(image_pil, prefix="detection"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(OUTPUT_DIR, f"{prefix}_{timestamp}.jpg")
    image_pil.save(path, quality=90)
    return path


def estimate_location_from_box(
    base_lat,
    base_lon,
    center_x,
    center_y,
    image_width,
    image_height
):
    """
    Simple approximation for demo purposes.

    A real system would use drone GPS, altitude, camera intrinsics,
    gimbal angle and map projection.

    Here we slightly offset the map pin based on where the object
    appears in the frame.
    """
    lat_offset = ((image_height / 2) - center_y) / image_height * 0.0012
    lon_offset = (center_x - (image_width / 2)) / image_width * 0.0012

    return base_lat + lat_offset, base_lon + lon_offset


# =====================================================
# STYLING
# =====================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #080c20;
    }

    .block-container {
        padding-top: 1.5rem;
    }

    h1, h2, h3 {
        color: #ffffff;
    }

    .sarathi-card {
        background: #11172d;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #273154;
    }

    .small-muted {
        color: #aeb8d6;
        font-size: 0.88rem;
    }

    .warning-note {
        background: #2c2138;
        color: #ffd166;
        padding: 0.75rem;
        border-radius: 10px;
        border: 1px solid #ffd166;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# MAIN APP
# =====================================================

init_db()

st.title(f"🚁 {APP_TITLE}")
st.caption(APP_SUBTITLE)

st.markdown(
    """
    <div class="warning-note">
    <b>Prototype notice:</b> SARathi is an academic MVP. 
    It is not a certified emergency-response product. 
    AI detections must be verified by a human operator before any action.
    </div>
    """,
    unsafe_allow_html=True
)

tabs = st.tabs(
    [
        "1. Detect from Image/Video",
        "2. Incident Map",
        "3. Manual SOS Report",
        "4. Incident Queue & Export",
        "5. Demo Guide"
    ]
)


# =====================================================
# TAB 1: DETECTION
# =====================================================

with tabs[0]:
    st.subheader("AI-assisted detection")

    left, right = st.columns([1.2, 1])

    with left:
        uploaded_file = st.file_uploader(
            "Upload a drone image or video",
            type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"]
        )

        source_type = st.radio(
            "Input type",
            ["Image", "Video"],
            horizontal=True
        )

        confidence_threshold = st.slider(
            "Detection confidence threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.35,
            step=0.05
        )

    with right:
        st.markdown("#### Base GPS location")
        st.caption(
            "For demo, enter the approximate drone/take-off location. "
            "Object pins are estimated around this point."
        )

        base_lat = st.number_input(
            "Latitude",
            value=19.0760,
            format="%.6f"
        )

        base_lon = st.number_input(
            "Longitude",
            value=72.8777,
            format="%.6f"
        )

        operator_note = st.text_input(
            "Operator note",
            value="AI-generated candidate; requires verification"
        )

    if uploaded_file is not None:
        if source_type == "Image":
            image = Image.open(uploaded_file).convert("RGB")

            st.markdown("### Uploaded frame")

            col_a, col_b = st.columns(2)

            with col_a:
                st.image(image, caption="Original image", use_column_width=True)

            with st.spinner("Running YOLO detection..."):
                detections, annotated = detect_objects(
                    image,
                    confidence_threshold
                )

            with col_b:
                st.image(
                   annotated,
                   caption="Annotated detection result",
                   use_column_width=True
               )
                

            annotated_path = save_image(annotated, prefix="image_detection")

            if len(detections) == 0:
                st.warning(
                    "No person or obstruction candidate found at this threshold."
                )
            else:
                st.success(f"{len(detections)} candidate detection(s) found.")

                detection_df = pd.DataFrame(detections)
                st.dataframe(
                    detection_df[["label", "confidence", "bbox"]],
                    use_container_width=True
                )

                st.markdown("### Operator action")

                for idx, det in enumerate(detections):
                    est_lat, est_lon = estimate_location_from_box(
                        base_lat,
                        base_lon,
                        det["center_x"],
                        det["center_y"],
                        image.width,
                        image.height
                    )

                    with st.expander(
                        f"{idx + 1}. {det['label']} "
                        f"confidence={det['confidence']:.2f}"
                    ):
                        c1, c2, c3 = st.columns([1, 1, 1])

                        corrected_lat = c1.number_input(
                            "Corrected latitude",
                            value=float(est_lat),
                            format="%.6f",
                            key=f"lat_img_{idx}"
                        )

                        corrected_lon = c2.number_input(
                            "Corrected longitude",
                            value=float(est_lon),
                            format="%.6f",
                            key=f"lon_img_{idx}"
                        )

                        status = c3.selectbox(
                            "Initial status",
                            ["New", "Verified", "Assigned"],
                            key=f"status_img_{idx}"
                        )

                        if st.button(
                            f"Create incident #{idx + 1}",
                            key=f"create_img_{idx}"
                        ):
                            add_incident(
                                source="Image upload",
                                label=det["label"],
                                confidence=det["confidence"],
                                latitude=corrected_lat,
                                longitude=corrected_lon,
                                status=status,
                                notes=operator_note,
                                image_path=annotated_path
                            )

                            st.success("Incident added to local queue.")

        else:
            st.markdown("### Video processing")

            sample_every_n_frames = st.slider(
                "Sample every N frames",
                min_value=5,
                max_value=120,
                value=30,
                step=5
            )

            max_sampled_frames = st.slider(
                "Maximum sampled frames",
                min_value=5,
                max_value=50,
                value=15,
                step=5
            )

            process_video = st.button("Process uploaded video")

            if process_video:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ) as tmp:
                    tmp.write(uploaded_file.read())
                    video_path = tmp.name

                cap = cv2.VideoCapture(video_path)

                if not cap.isOpened():
                    st.error("Could not open video.")
                else:
                    detections_all = []
                    sampled_count = 0
                    frame_index = 0
                    last_annotated = None

                    progress = st.progress(0)

                    while cap.isOpened():
                        ret, frame = cap.read()

                        if not ret:
                            break

                        if frame_index % sample_every_n_frames == 0:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            image = Image.fromarray(frame_rgb)

                            detections, annotated = detect_objects(
                                image,
                                confidence_threshold
                            )

                            last_annotated = annotated

                            for det in detections:
                                est_lat, est_lon = estimate_location_from_box(
                                    base_lat,
                                    base_lon,
                                    det["center_x"],
                                    det["center_y"],
                                    image.width,
                                    image.height
                                )

                                detections_all.append(
                                    {
                                        "frame": frame_index,
                                        "label": det["label"],
                                        "confidence": det["confidence"],
                                        "latitude": est_lat,
                                        "longitude": est_lon,
                                        "bbox": det["bbox"]
                                    }
                                )

                            sampled_count += 1
                            progress.progress(
                                min(sampled_count / max_sampled_frames, 1.0)
                            )

                            if sampled_count >= max_sampled_frames:
                                break

                        frame_index += 1

                    cap.release()

                    if last_annotated is not None:
                        st.image(
                            annotated,
                            caption="Annotated detection result",
                            use_column_width=True
                        )

                    if len(detections_all) == 0:
                        st.warning(
                            "No candidate person or obstruction detected "
                            "in the sampled frames."
                        )
                    else:
                        st.success(
                            f"{len(detections_all)} candidate events found."
                        )

                        df = pd.DataFrame(detections_all)
                        st.dataframe(
                            df[
                                [
                                    "frame",
                                    "label",
                                    "confidence",
                                    "latitude",
                                    "longitude",
                                    "bbox"
                                ]
                            ],
                            use_container_width=True
                        )

                        annotated_path = ""
                        if last_annotated is not None:
                            annotated_path = save_image(
                                last_annotated,
                                prefix="video_detection"
                            )

                        st.markdown("### Create incidents from top candidates")

                        top_df = df.sort_values(
                            by="confidence",
                            ascending=False
                        ).head(10)

                        for row_idx, row in top_df.iterrows():
                            with st.expander(
                                f"Frame {int(row['frame'])}: "
                                f"{row['label']} "
                                f"confidence={row['confidence']:.2f}"
                            ):
                                c1, c2, c3 = st.columns([1, 1, 1])

                                corrected_lat = c1.number_input(
                                    "Corrected latitude",
                                    value=float(row["latitude"]),
                                    format="%.6f",
                                    key=f"lat_vid_{row_idx}"
                                )

                                corrected_lon = c2.number_input(
                                    "Corrected longitude",
                                    value=float(row["longitude"]),
                                    format="%.6f",
                                    key=f"lon_vid_{row_idx}"
                                )

                                status = c3.selectbox(
                                    "Initial status",
                                    ["New", "Verified", "Assigned"],
                                    key=f"status_vid_{row_idx}"
                                )

                                if st.button(
                                    f"Create incident from frame {int(row['frame'])}",
                                    key=f"create_vid_{row_idx}"
                                ):
                                    add_incident(
                                        source=f"Video frame {int(row['frame'])}",
                                        label=row["label"],
                                        confidence=row["confidence"],
                                        latitude=corrected_lat,
                                        longitude=corrected_lon,
                                        status=status,
                                        notes=operator_note,
                                        image_path=annotated_path
                                    )

                                    st.success(
                                        "Incident added to local queue."
                                    )


# =====================================================
# TAB 2: MAP
# =====================================================

with tabs[1]:
    st.subheader("Local incident map")

    incidents = get_incidents()

    if incidents.empty:
        st.info(
            "No incidents yet. Create an AI-assisted detection or manual SOS report."
        )
    else:
        center_lat = incidents["latitude"].mean()
        center_lon = incidents["longitude"].mean()

        fmap = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=14,
            tiles="OpenStreetMap"
        )

        status_colours = {
            "New": "red",
            "Verified": "orange",
            "Assigned": "blue",
            "Closed": "green",
            "Rejected": "gray"
        }

        for _, row in incidents.iterrows():
            colour = status_colours.get(row["status"], "red")

            popup_html = f"""
            <b>Incident #{row['id']}</b><br>
            Label: {row['label']}<br>
            Confidence: {row['confidence']:.2f}<br>
            Status: {row['status']}<br>
            Source: {row['source']}<br>
            Time: {row['created_at']}<br>
            Notes: {row['notes']}
            """

            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                popup=popup_html,
                tooltip=f"#{row['id']} {row['label']} [{row['status']}]",
                icon=folium.Icon(
                    color=colour,
                    icon="info-sign"
                )
            ).add_to(fmap)

        st_folium(
            fmap,
            width=1200,
            height=540
        )

        st.markdown("### Update incident status")

        selected_id = st.selectbox(
            "Select incident",
            incidents["id"].tolist()
        )

        new_status = st.selectbox(
            "New status",
            ["New", "Verified", "Assigned", "Closed", "Rejected"]
        )

        if st.button("Update status"):
            update_incident_status(selected_id, new_status)
            st.success("Status updated. Refreshing...")
            time.sleep(0.5)
            st.rerun()


# =====================================================
# TAB 3: MANUAL SOS
# =====================================================

with tabs[2]:
    st.subheader("Manual SOS report")

    c1, c2 = st.columns(2)

    with c1:
        report_label = st.selectbox(
            "Report type",
            [
                "Manual SOS: Person stranded",
                "Manual SOS: Medical help needed",
                "Manual SOS: Blocked road",
                "Manual SOS: Food/water required",
                "Manual SOS: Other"
            ]
        )

        report_lat = st.number_input(
            "Incident latitude",
            value=19.0760,
            format="%.6f",
            key="manual_lat"
        )

        report_lon = st.number_input(
            "Incident longitude",
            value=72.8777,
            format="%.6f",
            key="manual_lon"
        )

    with c2:
        report_status = st.selectbox(
            "Initial status",
            ["New", "Verified", "Assigned"],
            key="manual_status"
        )

        report_notes = st.text_area(
            "Notes",
            value="Reported by responder phone / field volunteer."
        )

    if st.button("Add manual SOS report"):
        add_incident(
            source="Manual report",
            label=report_label,
            confidence=1.0,
            latitude=report_lat,
            longitude=report_lon,
            status=report_status,
            notes=report_notes,
            image_path=""
        )

        st.success("Manual report added to incident queue.")


# =====================================================
# TAB 4: QUEUE AND EXPORT
# =====================================================

with tabs[3]:
    st.subheader("Incident queue")

    incidents = get_incidents()

    if incidents.empty:
        st.info("No incidents recorded.")
    else:
        st.dataframe(
            incidents,
            use_container_width=True
        )

        csv = incidents.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download incident queue as CSV",
            data=csv,
            file_name="sarathi_incident_queue.csv",
            mime="text/csv"
        )

        with st.expander("Danger zone"):
            st.warning(
                "This deletes all local incident records from the SQLite database."
            )

            if st.button("Delete all incidents"):
                delete_all_incidents()
                st.success("All incidents deleted.")
                time.sleep(0.5)
                st.rerun()


# =====================================================
# TAB 5: DEMO GUIDE
# =====================================================

with tabs[4]:
    st.subheader("How to demonstrate this prototype")

    st.markdown(
        """
        ### Suggested 3-minute demo flow

        **Step 1 — Start local command centre**  
        Run the Streamlit app on your laptop.

        **Step 2 — Upload drone or phone footage**  
        Use either:
        - A short drone clip,
        - A phone video from a terrace/open area,
        - Or an openly licensed flood/rescue image.

        **Step 3 — Run AI detection**  
        The app identifies candidate people and possible obstructions.

        **Step 4 — Human verification**  
        The operator checks the detection, corrects the location if needed, 
        and creates a verified incident.

        **Step 5 — Incident map**  
        Open the map tab and show the new incident pin.

        **Step 6 — Assignment workflow**  
        Change the status from `New` to `Verified` or `Assigned`.

        **Step 7 — Export**  
        Download the CSV incident queue as evidence of auditability.

        ---

        ### What this prototype proves

        - Edge AI can reduce manual video scanning.
        - Human verification prevents unsafe autonomous dispatch.
        - Incidents can be logged and mapped without a complex cloud backend.
        - The same architecture can later be extended to drones, phones, and Azure sync.

        ---

        ### What this prototype does not claim

        - It is not a certified emergency-response system.
        - It does not replace rescue commanders.
        - It does not perform medical diagnosis.
        - It does not guarantee detection in real disaster conditions.
        """
    )