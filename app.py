import os
import tempfile
import subprocess

import cv2
import streamlit as st
from ultralytics import YOLO
import imageio_ffmpeg

st.set_page_config(
    page_title="License Plate Detection",
    page_icon="🚘",
    layout="wide"
)

st.title("🚘 License Plate Detection System")
st.write("Detect vehicle license plates in images and videos using YOLO.")

MODEL_PATH = "best.pt"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file 'best.pt' not found. Please upload it with app.py.")
        st.stop()

    return YOLO(MODEL_PATH)


model = load_model()

confidence = st.slider(
    "Detection Confidence",
    min_value=0.05,
    max_value=0.90,
    value=0.25,
    step=0.05
)

image_size = st.selectbox(
    "Image Size",
    [640, 960, 1280],
    index=0
)

image_tab, video_tab = st.tabs(["🖼️ Image Detection", "🎥 Video Detection"])


def detect_image(uploaded_file):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
        temp.write(uploaded_file.getvalue())
        image_path = temp.name

    try:
        result = model.predict(
            source=image_path,
            conf=confidence,
            imgsz=image_size,
            verbose=False
        )[0]

        annotated_image = cv2.cvtColor(
            result.plot(),
            cv2.COLOR_BGR2RGB
        )

        return annotated_image, len(result.boxes)

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)


def detect_video(uploaded_file):
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "input_video.mp4")
        raw_path = os.path.join(temp_dir, "raw_output.mp4")
        output_path = os.path.join(temp_dir, "detected_video.mp4")

        with open(input_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise ValueError("Could not open the uploaded video.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            fps = 25.0

        writer = cv2.VideoWriter(
            raw_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Could not create the output video.")

        progress = st.progress(0)
        processed = 0

        try:
            while True:
                success, frame = cap.read()

                if not success:
                    break

                result = model.predict(
                    source=frame,
                    conf=confidence,
                    imgsz=image_size,
                    verbose=False
                )[0]

                writer.write(result.plot())
                processed += 1

                if total_frames > 0:
                    progress.progress(
                        min(processed / total_frames, 1.0)
                    )

        finally:
            cap.release()
            writer.release()

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        conversion = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel", "error",
                "-i", raw_path,
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_path
            ],
            capture_output=True,
            text=True
        )

        if conversion.returncode != 0:
            raise RuntimeError("Could not convert the processed video.")

        with open(output_path, "rb") as f:
            return f.read()


with image_tab:
    image_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if image_file is not None:
        st.image(
            image_file,
            caption="Original Image",
            use_container_width=True
        )

        if st.button("Detect License Plates", key="detect_image"):
            with st.spinner("Detecting license plates..."):
                try:
                    result_image, count = detect_image(image_file)

                    st.image(
                        result_image,
                        caption="Detection Result",
                        use_container_width=True
                    )

                    st.success(f"Detected objects: {count}")

                except Exception as error:
                    st.error(f"Detection failed: {error}")


with video_tab:
    video_file = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if video_file is not None:
        st.video(video_file)

        if st.button("Detect in Video", key="detect_video"):
            try:
                with st.spinner("Processing video. Please wait..."):
                    video_bytes = detect_video(video_file)

                st.success("Video processing completed!")

                st.video(video_bytes)

                st.download_button(
                    label="Download Detected Video",
                    data=video_bytes,
                    file_name="license_plate_detection.mp4",
                    mime="video/mp4"
                )

            except Exception as error:
                st.error(f"Video processing failed: {error}")


st.markdown("---")
st.caption("License Plate Detection | Powered by YOLO, OpenCV and Streamlit")
