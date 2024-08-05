import streamlit as st
import tensorflow as tf
import numpy as np

#Model Prediction

def m_pred(test_image):
    model=tf.keras.models.load_model('trained_model.keras')
    image=tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])
    prediction=model.predict(input_arr)
    result=np.argmax(prediction)
    return result

# app
st.header("PLANTGUARD")




# Home
def home_page():
    st.header("Welcome to PlantGuard: Your Partner in Plant Health")
    img="disease-image.jpg"
    st.image(img,use_column_width=True)
    

    markdown_content="""
    At PlantGuard, we utilize cutting-edge machine learning technology to help you detect diseases in plant leaves quickly and accurately. Whether you are a home gardener, botanist, or agricultural enthusiast, our platform is designed to assist you in maintaining the health and vitality of your plants.

    ## How It Works

    1. **Upload an Image**: Take a clear photo of the plant leaf and upload it to our platform.
    2. **Analyze**: Our advanced machine learning algorithms will analyze the image to identify any signs of disease.
    3. **Get Results**: Receive instant feedback on the health of the leaf, including the type of disease and recommended treatment options.

    ## Our Mission

    Our mission is to empower the plant care community with innovative technology solutions that promote healthy and sustainable plant growth. By providing a quick and accurate leaf disease detection tool, we aim to minimize plant damage, reduce the use of unnecessary chemicals, and enhance overall plant care.

    ## Why Choose PlantGuard?

    - **Accuracy**: Our state-of-the-art ML models are trained on an extensive dataset of leaf images, ensuring high accuracy in disease detection.
    - **Ease of Use**: Designed with simplicity in mind, our platform is intuitive and easy to navigate.
    - **Quick Results**: Get instant analysis and recommendations, allowing you to act fast and save your plants.
    - **Support**: Our team of experts is always here to help with any questions or concerns you may have.

    ## Join the Community

    Become a part of the PlantGuard community and contribute to a healthier, more sustainable future for plant care. Sign up today to start protecting your plants and ensuring their health.
    """
    st.markdown(markdown_content)
    
# about us


    #Prediction

def detection_page():
    st.header("Disease Detection")
    test_img=st.file_uploader("Choose a Plant Image:")
    if(st.button("Show Image")):
        st.image(test_img,use_column_width=True )
    if(st.button("Predict")):
        st.snow()
        st.write("Result:")
        result_index=m_pred(test_img)
        class_name=['Apple___Apple_scab',
        'Apple___Black_rot',
        'Apple___Cedar_apple_rust',
        'Apple___healthy',
        'Blueberry___healthy',
        'Cherry_(including_sour)___Powdery_mildew',
        'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
        'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight',
        'Corn_(maize)___healthy',
        'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)',
        'Peach___Bacterial_spot',
        'Peach___healthy',
        'Pepper,_bell___Bacterial_spot',
        'Pepper,_bell___healthy',
        'Potato___Early_blight',
        'Potato___Late_blight',
        'Potato___healthy',
        'Raspberry___healthy',
        'Soybean___healthy',
        'Squash___Powdery_mildew',
        'Strawberry___Leaf_scorch',
        'Strawberry___healthy',
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
        'Tomato___Late_blight',
        'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy']
        st.success(f"Predicted Disease: {class_name[result_index]}")


def about_page():
    st.header("About")
    st.markdown("""
    Discover the future of plant care with PlantGuard, where we blend advanced technology and a passion for botany to create innovative solutions for detecting diseases in plant leaves.

## Who We Are

PlantGuard is a team of passionate plant enthusiasts, data scientists, and technology experts dedicated to improving plant care through innovative solutions. We believe that healthy plants are the foundation of a sustainable and thriving environment, and our goal is to empower everyone—from home gardeners to professional botanists—with the tools they need to keep their plants healthy.

## Our Technology

At the core of PlantGuard is our advanced machine learning algorithm, meticulously trained on a vast dataset of plant leaf images. This technology allows us to accurately identify a wide range of plant diseases, providing you with precise and actionable insights. Our platform is designed to be simple and intuitive, making plant care accessible to everyone, regardless of their level of expertise.

## About Our Dataset

This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this github repo. This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.
## Our Vision

Our vision is to create a world where every plant can thrive. By providing an easy-to-use and accurate disease detection tool, we aim to reduce plant damage, minimize the use of harmful chemicals, and promote sustainable plant care practices. We are committed to continuous improvement, constantly updating our technology and expanding our dataset to ensure the highest level of accuracy and reliability.

## Why Choose Us

- **Expertise**: Our team combines deep knowledge of plant biology with cutting-edge machine learning techniques.
- **Innovation**: We are at the forefront of technology, continually advancing our algorithms to provide the best possible service.
- **Community**: We believe in the power of community and strive to support and educate our users in their plant care journey.
- **Sustainability**: Our focus is on promoting healthy and sustainable plant care practices that benefit both you and the environment.

## Join Us

Become part of the PlantGuard community and take the first step towards healthier plants. Together, we can create a greener and more sustainable future.

Thank you for choosing PlantGuard as your trusted partner in plant health.
                """)
    
st.sidebar.title("Dashboard")
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Sidebar buttons for navigation
if st.sidebar.button("Home"):
    st.session_state.page = 'Home'
if st.sidebar.button("Disease Detection"):
    st.session_state.page = 'Disease Detection'
if st.sidebar.button("About Us"):
    st.session_state.page = 'About Us'

# Display the selected page
if st.session_state.page == 'Home':
    home_page()
elif st.session_state.page == 'Disease Detection':
    detection_page()
elif st.session_state.page == 'About Us':
    about_page()
