import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load Model
model = load_model("face_shape.keras")

# Class names (same order as training)
class_names = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

st.title("Face Shape Detector")





st.markdown("""
<style>

/* Main Background */
.stApp{
    background: linear-gradient(to right,#F8F5F2,#E8D9E6);
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#D8BFD8;
}

/* Title */
h1{
    color:#5A3E5C;
    text-align:center;
    font-size:42px;
}

/* Upload Box */
[data-testid="stFileUploader"]{
    border:2px dashed #8E6C88;
    border-radius:15px;
    padding:10px;
}

/* Success Box */
div[data-testid="stAlert"]{
    border-radius:15px;
}

/* Buttons */
.stButton>button{
    width:100%;
    background:#8E6C88;
    color:white;
    border-radius:12px;
    font-weight:bold;
    border:none;
    height:45px;
}

.stButton>button:hover{
    background:#6E4F71;
}

/* Camera Box */
[data-testid="stCameraInput"]{
    border:2px solid #8E6C88;
    border-radius:15px;
    padding:10px;
}

/* Footer */
.footer{
text-align:center;
color:gray;
padding-top:20px;
}

</style>
""",unsafe_allow_html=True)



st.markdown("""
<h1>🧠 AI Face Shape Detector</h1>
""",unsafe_allow_html=True)

st.write(
"Upload an image or capture using your camera to detect your face shape instantly."
)




st.sidebar.image("face_shape.png", use_container_width=True)

st.sidebar.markdown("## Face Shapes")

st.sidebar.write("""
✔ Heart

✔ Oval

✔ Round

✔ Square

✔ Oblong
""")



# Upload Image
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.image(uploaded_file, width=250)
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224,224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"Face Shape : {class_names[predicted_class]}")

    st.write(f"Confidence : {confidence*100:.2f}%")


    # Camera State
if "open_camera" not in st.session_state:
    st.session_state.open_camera = False

col1, col2 = st.columns(2)

with col1:
    if st.button("Open Camera"):
        st.session_state.open_camera = True

with col2:
    if st.button("Close Camera"):
        st.session_state.open_camera = False

if st.session_state.open_camera:

    cam = st.camera_input("Capture Face")

    if cam is not None:

        img = Image.open(cam).convert("RGB")
        st.image(img, width=250)
        img = img.resize((224,224))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        st.markdown(f"""
       <div style="
        background:#FFF8FC;
        padding:20px;
        border-radius:15px;
        border-left:8px solid #8E6C88;
       ">

        <h3 style="color:#5A3E5C;">
            Predicted Face Shape
       </h3>

        <h2 style="color:#8E6C88;">
            {class_names[predicted_class]}
        </h2>

        <h4>
            Confidence : {confidence*100:.2f}%
        </h4>

        </div>
        """,unsafe_allow_html=True)