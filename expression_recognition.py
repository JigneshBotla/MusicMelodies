import cv2
from deepface import DeepFace
import webbrowser
import time

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

# Start capturing video
cap = cv2.VideoCapture(0)

# Initialize emotion count
emotion_count = {'happy': 0, 'angry': 0, 'sad': 0, 'surprise': 0, 'neutral': 0, 'fear': 0}

# Start timing the observation period
start_time = time.time()

while time.time() - start_time < 10:  # Observe for 10 seconds
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale to RGB for DeepFace
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = rgb_frame[y:y + h, x:x + w]

        # Analyze emotions
        try:
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
        except Exception as e:
            print(f"Error analyzing emotion: {e}")
            continue

        # Check if the emotion exists in the dictionary before incrementing
        if emotion in emotion_count:
            emotion_count[emotion] += 1
        else:
            print(f"Unknown emotion detected: {emotion}")

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Determine dominant emotion
dominant_emotion = max(emotion_count, key=emotion_count.get)
print(f"Dominant emotion detected: {dominant_emotion}")

# Define playlist URLs 
playlist_urls = {
    'happy': "https://open.spotify.com/search/happy/playlists",
    'angry': "https://open.spotify.com/search/angry/playlists",
    'sad': "https://open.spotify.com/search/sad/playlists",
    'surprise': "https://open.spotify.com/search/surprise/playlists",
    'neutral': "https://open.spotify.com/search/cool%20songs/playlists",
    'fear': "https://open.spotify.com/search/fear/playlists"
}

# Get the playlist URL for the dominant emotion
playlist_url = playlist_urls.get(dominant_emotion)

# Open the playlist URL in the default web browser
if playlist_url:
    webbrowser.open(playlist_url)

# Release resources
cap.release()
cv2.destroyAllWindows()
