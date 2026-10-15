import os
import tempfile
import subprocess

import cv2
import streamlit as st
from ultralytics import YOLO
import imageio_ffmpeg

st.set_page_config(page_title="License Plate Detection", page_icon="🚘", layout="wide")
st.title("🚘 License Plate Detection")
st.write("Upload an image or video to detect license plates.")

MODEL_PATH = "best.pt"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file 'best.pt' not found. Add best.pt to the same folder as app.py.")
        st.stop()
    return YOLO(MODEL_PATH)


model = load_model()


def detect_image(uploaded_file, confidence, image_size):
    suffix = os.path.splitext(uploaded_file.name)[1] or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        input_path = temp_file.name

    try:
        results = model.predict(
            source=input_path,
            conf=confidence,
            imgsz=int(image_size),
            verbose=False
        )
        annotated = results[0].plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        return annotated, len(results[0].boxes)
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


def detect_video(uploaded_file, confidence, image_size):
    input_suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "input" + input_suffix)
        raw_path = os.path.join(temp_dir, "annotated_raw.mp4")
        output_path = os.path.join(temp_dir, "annotated.mp4")

        with open(input_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Could not open this video. Please try another video file.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if not fps or fps <= 0:
            fps = 25.0
        if width <= 0 or height <= 0:
            cap.release()
            raise ValueError("Could not read the video dimensions.")

        writer = cv2.VideoWriter(
            raw_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Could not create the output video.")

        progress = st.progress(0, text="Processing video...")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                result = model.predict(
                    source=frame,
                    conf=confidence,
                    imgsz=int(image_size),
                    verbose=False
                )[0]
                writer.write(result.plot())
                processed += 1

                if frame_count > 0:
                    progress.progress(
                        min(processed / frame_count, 1.0),
                        text=f"Processing frame {processed} of {frame_count}..."
                    )
        finally:
            cap.release()
            writer.release()

        progress.empty()

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        conversion = subprocess.run(
            [
                ffmpeg_exe, "-y", "-loglevel", "error",
                "-i", raw_path,
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_path
            ],
            capture_output=True,
            text=True
        )

        if conversion.returncode != 0 or not os.path.exists(output_path):
            raise RuntimeError(
                "Video conversion failed. Try a different video. "
                + conversion.stderr[-500:]
            )

        with open(output_path, "rb") as f:
            return f.read()


confidence = st.slider(
    "Confidence threshold",
    min_value=0.05,
    max_value=0.90,
    value=0.25,
    step=0.05
)

image_size = st.selectbox(
    "Image size (higher can be slower)",
    options=[640, 960, 1280],
    index=0
)

image_tab, video_tab = st.tabs(["Detect in Image", "Detect in Video"])

with image_tab:
    image_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_upload"
    )

    if image_file is not None:
        st.image(image_file, caption="Original image", use_container_width=True)

        if st.button("Detect plates in image", type="primary"):
            with st.spinner("Detecting license plates..."):
                try:
                    annotated_image, count = detect_image(
                        image_file, confidence, image_size
                    )
                    st.image(
                        annotated_image,
                        caption="Detection result",
                        use_container_width=True
                    )
                    st.success(f"Detection complete. Detected objects: {count}")
                except Exception as exc:
                    st.error(f"Detection failed: {exc}")

with video_tab:
    video_file = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload"
    )

    if video_file is not None:
        st.video(video_file)

        if st.button("Detect plates in video", type="primary"):
            try:
                with st.spinner("Processing video. Long videos may take time..."):
                    output_video = detect_video(
                        video_file, confidence, image_size
                    )

                st.success("Video detection complete!")
                st.video(output_video)
                st.download_button(
                    "Download processed video",
                    data=output_video,
                    file_name="license_plate_detection.mp4",
                    mime="video/mp4"
                )
            except Exception as exc:
                st.error(f"Video processing failed: {exc}")

st.caption("Detection quality depends on the trained best.pt model.")
