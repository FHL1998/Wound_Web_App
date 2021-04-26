import streamlit as st
from multiapp import MultiApp
from apps import opencv_based_segmentation, ai_based_segmentation  # import your app modules here

app = MultiApp()

# Add all your application here
app.add_app("Wound Segmentation-Based on Opencv", opencv_based_segmentation.app)
app.add_app("Wound Segmentation-Based on Deep Learning", ai_based_segmentation.app)

# The main app
app.run()
