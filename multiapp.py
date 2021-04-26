"""Frameworks for running multiple Streamlit applications as a single app.
"""
import requests
import streamlit as st
from streamlit_lottie import st_lottie


class MultiApp:

    def __init__(self):
        self.apps = []



    def add_app(self, title, func):
        """Adds a new application.
        Parameters
        ----------
        func:
            the python function to render this app.
        title:
            title of the app. Appears in the dropdown in the sidebar.
        """

        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        st.set_page_config(
            page_title="FYP Web APP Wound Segmentation",
            # page_icon="random",
            page_icon="https://wound-1301658428.cos.ap-nanjing.myqcloud.com/image/hospital.png",
        )
        st.image("https://wound-1301658428.cos.ap-nanjing.myqcloud.com/image/hospital.png", width=100)

        with st.sidebar:
            st.info(
                "🎈 **NEW:** You can also check the [App Introduction Blog ](https://oliverfan.top/posts/2adb/) for "
                "more details! "
            )
        st.sidebar.header('🌈Navigate To')
        app = st.sidebar.radio(
            'Function',
            self.apps,
            format_func=lambda app: app['title'],
        )
        # st.balloons()
        st.markdown("---", unsafe_allow_html=True)
        st.sidebar.header('About')
        with st.sidebar.beta_expander("Acknowledgement"):
            st.write("""**Wound Segmentation** is a *web app* for estimating size and amount of tissue types in wounds based 
                                on the **FYP Project:Development of an AI-based region detection App for wound healing therapy**,which is the supplement 
                                of the Mobile App. Thanks Professor Fu and Professor Lu for their great help during the completion of the 
                                project.Thanks for the opportunity given by [NUS]( https://www.nus.edu.sg). """)
        with st.sidebar.beta_expander("Contact Author"):
            st.write("""
            ✔️Email:fhlielts8@gmail.com
            ✔️Mobile:+86 18810328618
            """)
            st.markdown('[![Premchandra Singh]\
                            (https://img.shields.io/badge/Author-@OliverFan-gray.svg?colorA=gray&colorB=dodgerblue&logo=github)]\
                            (https://github.com/TOESL100)')
        app['function']()
