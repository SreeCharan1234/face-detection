import cv2
import streamlit as st
import numpy as np
from deepface import DeepFace
import google.generativeai as genai
from streamlit_extras.let_it_rain import rain
import os
os.getenv("AIzaSyBIBVb-0Z0QwaucMGOGy8-j_RM22X-4-lE")
genai.configure(api_key="AIzaSyBIBVb-0Z0QwaucMGOGy8-j_RM22X-4-lE")
def example():
    rain(
        emoji="*",
        font_size=54,
        falling_speed=5,
        animation_length="infinite",
    )
# Function to analyze facial attributes using DeepFace
def analyze_frame(frame):
    result = DeepFace.analyze(img_path=frame, actions=['age', 'gender', 'race', 'emotion'],
                              enforce_detection=False,
                              detector_backend="opencv",
                              align=True,
                              silent=False)
    return result
def get_gemini_response(question):
    model = genai.GenerativeModel('gemini-pro')
    chat = model.start_chat(history=[])
    response = model.generate_content(question)
    return response.text
def overlay_text_on_frame(frame, texts):
    overlay = frame.copy()
    alpha = 0.9  # Adjust the transparency of the overlay
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    text_position = 15 # Where the first text is put intost,write the overlay
    
    return frame

def facesentiment():
    st.title("Real-Time Health Docotor ")
    example()
    col3,col4=st.columns([4,2])
    i=0
    with col3:
        questions = [
            
            "how are you feeling today?",
            "Do you have any chronic health conditions?",
            "Are you experiencing any pain or discomfort?",
            "Have you recently had any injuries or surgeries?"
        ]       
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "Hi 😊 Tell Me Your Name?"}]
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
            
        if prompt := st.chat_input():
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            msg=questions[i]
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg) 
            
    with col4:    # Create two columns: one for the video feed and one for the text details
        col1, col2 = st.columns(2)

        cap = cv2.VideoCapture(0)
        
        # Display the camera feed in the first column
        with col1:
            stframe = st.image([]) 

        # Placeholder for displaying the text information in the second column
        with col2:
            text_placeholder = st.empty()

        while True:
            ret, frame = cap.read()

            result = analyze_frame(frame)

            face_coordinates = result[0]["region"]
            x, y, w, h = face_coordinates['x'], face_coordinates['y'], face_coordinates['w'], face_coordinates['h']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{result[0]['dominant_emotion']}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create text details for the analysis
            texts = [
                f"Age: {result[0]['age']}",
                f"Face Confidence: {round(result[0]['face_confidence'], 3)}",
                f"Gender: {result[0]['dominant_gender']} {round(result[0]['gender'][result[0]['dominant_gender']], 3)}",
                f"Race: {result[0]['dominant_race']}",
                f"Dominant Emotion: {result[0]['dominant_emotion']} {round(result[0]['emotion'][result[0]['dominant_emotion']], 1)}",
            ]

            # Display the frame with overlay in the first column
            frame_with_overlay = overlay_text_on_frame(frame_rgb, texts)
            stframe.image(frame_with_overlay, channels="RGB")
            
            # Update the text information in the second column
            text_placeholder.write("\n".join(texts))

        cap.release()
        cv2.destroyAllWindows()

def main():  
    facesentiment()
    

if __name__ == "__main__":
    main()
