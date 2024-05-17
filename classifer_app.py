import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Load model once at startup
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('models/my_model.keras')

model = load_model()

# Prediction function
def model_prediction(image):
    image = tf.keras.preprocessing.image.load_img(image, target_size=(256, 256))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return predictions

# Define the class names globallys
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# Descriptions for each disease
DISEASE_DESCRIPTIONS = {
    'Pepper__bell___Bacterial_spot': "Bacterial spot is a common disease affecting pepper plants, causing small, water-soaked lesions on leaves and fruit.",
    'Pepper__bell___healthy': "The plant is healthy and shows no signs of disease.",
    'Potato___Early_blight': "Early blight is a fungal disease that causes leaf spots and can lead to significant yield loss.",
    'Potato___Late_blight': "Late blight is a serious fungal disease that affects both the foliage and tubers of potato plants.",
    'Potato___healthy': "The plant is healthy and shows no signs of disease.",
    'Tomato_Bacterial_spot': "Bacterial spot on tomatoes causes lesions on leaves, stems, and fruit, leading to reduced crop quality.",
    'Tomato_Early_blight': "Early blight in tomatoes causes concentric ring spots on leaves and fruit, leading to premature defoliation.",
    'Tomato_Late_blight': "Late blight is a destructive fungal disease causing dark, water-soaked lesions on leaves and fruit.",
    'Tomato_Leaf_Mold': "Leaf mold causes yellow spots on the upper leaf surface and velvety, olive-green mold on the underside.",
    'Tomato_Septoria_leaf_spot': "Septoria leaf spot causes numerous small, circular spots on leaves, leading to premature leaf drop.",
    'Tomato_Spider_mites_Two_spotted_spider_mite': "Two-spotted spider mites cause stippling and discoloration of leaves, leading to reduced plant vigor.",
    'Tomato__Target_Spot': "Target spot causes dark, concentric lesions on leaves and stems, affecting fruit quality.",
    'Tomato__Tomato_YellowLeaf__Curl_Virus': "Tomato Yellow Leaf Curl Virus causes yellowing and curling of leaves, stunting plant growth.",
    'Tomato__Tomato_mosaic_virus': "Tomato mosaic virus causes mottled, discolored leaves and can lead to reduced fruit yield and quality.",
    'Tomato_healthy': "The plant is healthy and shows no signs of disease."
}

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition", "Image Processing"])

# Home Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help identify plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Additional Features
    - **Image Processing:** Visit the **Image Processing** page to explore various techniques such as grayscale conversion, resizing, edge detection, and more. These tools help enhance and analyze plant images before disease recognition.
    - **Tools Used:** Our system is built using advanced tools and technologies including Streamlit for the web application, TensorFlow for machine learning, OpenCV for image processing, and PIL for image handling.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)


# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    ### About the Plant Disease Recognition System

    The Plant Disease Recognition System is designed to help farmers and gardeners identify diseases in their crops quickly and accurately. Utilizing state-of-the-art machine learning algorithms, our system analyzes uploaded images of plants and detects potential diseases, providing users with detailed information and recommendations for treatment.

    #### Our Team
    This project is developed by a team of first-year Master's students at ENSET Mohammedia:

    - **Eljaafari Ossama**: Responsible for data preprocessing and model training.
    - **AITSIRIR Tarik**: Focuses on the web application development and integration.
    - **ISMAOUI Rabii**: Specializes in image processing techniques and model optimization.

    #### About Dataset
    The dataset used in this project consists of approximately 87,000 RGB images of healthy and diseased crop leaves, categorized into 38 different classes. The images are divided into training, validation, and test sets in an 80/20 ratio.

    - **Train:** 70,295 images
    - **Validation:** 17,572 images
    - **Test:** 33 images

    The dataset was augmented offline to ensure robustness and variety.

    #### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system processes the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    For more information or to get started, visit the **Disease Recognition** page.
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    if st.button("Show Image"):
        st.image(test_image, use_column_width=True)
    if test_image is not None:
        if st.button("Predict"):
            with st.spinner("Processing..."):
                predictions = model_prediction(test_image)
                result_index = np.argmax(predictions)
                result_confidence = predictions[0][result_index]
                disease_name = CLASS_NAMES[result_index]
                st.success(f"Model is predicting it's a {disease_name} with confidence {result_confidence:.2f}")
                st.markdown(f"### Disease Description\n{DISEASE_DESCRIPTIONS[disease_name]}")
                st.balloons()

# Image Processing Page
elif app_mode == "Image Processing":
    st.header("Image Processing Techniques")
    st.markdown("### Choose an image to apply different image processing techniques:")

    img_file = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    
    if img_file is not None:
        image = Image.open(img_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Convert image to numpy array
        img_array = np.array(image)

        # Technique selection
        technique = st.selectbox("Select a technique:", [
            "Grayscale", "Resize", "Edge Detection", 
            "Filtrage spatial", "Ajustement du contraste", 
            "Détection de contours", "Segmentation"
        ])

        if technique == "Grayscale":
            st.markdown("#### Grayscale Conversion")
            st.markdown("La conversion en niveaux de gris est effectuée en utilisant la formule :")
            st.latex(r'Y = 0.299 \times R + 0.587 \times G + 0.114 \times B')
            gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            st.image(gray_image, caption='Grayscale Image', use_column_width=True)

        elif technique == "Resize":
            st.markdown("#### Resize")
            st.markdown("L'image est redimensionnée aux dimensions spécifiées en utilisant l'interpolation.")
            width = st.number_input("Width:", value=256)
            height = st.number_input("Height:", value=256)
            resized_image = cv2.resize(img_array, (width, height))
            st.image(resized_image, caption='Resized Image', use_column_width=True)

        elif technique == "Edge Detection":
            st.markdown("#### Edge Detection (Canny)")
            st.markdown("Le détecteur de contours de Canny est utilisé pour trouver les contours dans l'image.")
            edges = cv2.Canny(img_array, 100, 200)
            st.image(edges, caption='Edge Detection Image', use_column_width=True)

        elif technique == "Filtrage spatial":
            st.markdown("#### Filtrage spatial (Gaussian Blur)")
            st.markdown("Le flou gaussien est une technique de filtrage spatial qui consiste à remplacer chaque pixel de l'image par une moyenne pondérée de ses voisins, selon la distribution gaussienne.")
            st.latex(r'G(x, y) = \frac{1}{2\pi\sigma^2}e^{-\frac{x^2 + y^2}{2\sigma^2}}')
            st.markdown("Où :")
            st.markdown("- $G(x, y)$ est la valeur de l'image floutée au point $(x, y)$.")
            st.markdown("- $\sigma$ est l'écart-type du noyau gaussien.")
            blurred_image = cv2.GaussianBlur(img_array, (5, 5), 0)
            st.image(blurred_image, caption='Gaussian Blurred Image', use_column_width=True)
                
        elif technique == "Ajustement du contraste":
            st.markdown("#### Contrast Adjustment (Gamma Correction)")
            st.markdown("La correction gamma est appliquée pour ajuster la luminosité et le contraste de l'image.")
            st.latex(r'I_{\text{corrected}} = 255 \times \left(\frac{I_{\text{original}}}{255}\right)^\gamma')
            gamma = st.slider("Gamma value", 0.1, 3.0, 1.0)
            gamma_corrected = np.array(255 * (img_array / 255) ** gamma, dtype='uint8')
            st.image(gamma_corrected, caption='Gamma Corrected Image', use_column_width=True)

        elif technique == "Détection de contours":
            st.markdown("#### Edge Detection (Prewitt)")
            st.markdown("Le détecteur de contours de Prewitt est utilisé pour trouver les contours dans l'image.")
            kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
            kernely = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
            prewittx = cv2.filter2D(img_array, -1, kernelx)
            prewitty = cv2.filter2D(img_array, -1, kernely)
            prewitt_image = prewittx + prewitty
            st.image(prewitt_image, caption='Prewitt Edge Detection Image', use_column_width=True)

        elif technique == "Segmentation":
                    st.markdown("#### Segmentation (K-means Clustering)")
                    st.markdown("Le clustering K-means est une méthode non supervisée utilisée pour regrouper des données similaires ensemble.")
                    st.markdown("L'algorithme fonctionne en itérant entre deux étapes :")
                    st.markdown("1. **Attribution de clusters :** Chaque point de données est attribué au cluster dont le centre est le plus proche.")
                    st.markdown("2. **Mise à jour des centres des clusters :** Les centres de chaque cluster sont mis à jour en prenant la moyenne de tous les points attribués au cluster.")
                    st.latex(r"\underset{\mathbf{c}}{\min} \sum_{i=1}^{n} \left\| \mathbf{x}_i - \mathbf{\mu}_{c_i} \right\|^2")
                    st.markdown("Où :")
                    st.markdown("- $n$ est le nombre de points de données.")
                    st.markdown("- $\mathbf{x}_i$ est le i-ème point de données.")
                    st.markdown("- $\mathbf{c}$ est l'ensemble des clusters.")
                    st.markdown("- $\mathbf{\mu}_{c_i}$ est le centre du cluster auquel le point de données $\mathbf{x}_i$ est attribué.")
                    st.markdown("Après un certain nombre d'itérations, les centres des clusters convergent vers des positions stables, donnant des regroupements finaux.")
                    
                    Z = img_array.reshape((-1, 3))
                    Z = np.float32(Z)
                    K = st.slider("Number of clusters (K)", 2, 10, 4)
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                    center = np.uint8(center)
                    res = center[label.flatten()]
                    segmented_image = res.reshape((img_array.shape))
                    st.image(segmented_image, caption='K-means Clustering Image', use_column_width=True)

else:
    st.error("Please select a page to proceed.")
