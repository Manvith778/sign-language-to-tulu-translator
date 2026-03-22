from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

# Gesture-to-language mapping
GESTURE_TO_LANGUAGE = {
    "thumbs_up": {"kannada": "ಹೌದು", "tulu": "ಅಂದ್"},
    "thumbs_down": {"kannada": "ಇಲ್ಲ", "tulu": "ಇಜ್ಜ"},
    "peace": {"kannada": "ಶಾಂತಿ", "tulu": "ಶಾಂತಿ"},
    "ok_sign": {"kannada": "ಸರಿ", "tulu": "ಆವು"},
}

# Normalize landmarks for different resolutions
def normalize_landmarks(landmarks, image_width, image_height):
    return [(lm.x * image_width, lm.y * image_height) for lm in landmarks]


def detect_gesture(landmarks):
    try:
        # Extracting landmark coordinates
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        thumb_tip = landmarks[4]
        
        index_base = landmarks[5]
        middle_base = landmarks[9]
        ring_base = landmarks[13]
        pinky_base = landmarks[17]
        thumb_base = landmarks[2]

        # Helper function to check if a finger is raised (extended)
        def is_finger_up(tip, base):
            return tip[1] < base[1]  # Y-axis lower means finger is extended

        # Helper function to check if a finger is down (curled)
        def is_finger_down(tip, base):
            return tip[1] > base[1]  # Y-axis higher means finger is curled

        # Thumbs Up: Thumb up and other fingers down
        if is_finger_up(thumb_tip, thumb_base) and \
           is_finger_down(index_tip, index_base) and \
           is_finger_down(middle_tip, middle_base) and \
           is_finger_down(ring_tip, ring_base) and \
           is_finger_down(pinky_tip, pinky_base):
            return "thumbs_up"

        # Thumbs Down: Thumb down and other fingers down
        elif is_finger_down(thumb_tip, thumb_base) and \
             is_finger_down(index_tip, index_base) and \
             is_finger_down(middle_tip, middle_base) and \
             is_finger_down(ring_tip, ring_base) and \
             is_finger_down(pinky_tip, pinky_base):
            return "thumbs_down"

        # Peace Sign: Index and middle up, rest down
        elif is_finger_up(index_tip, index_base) and \
             is_finger_up(middle_tip, middle_base) and \
             is_finger_down(ring_tip, ring_base) and \
             is_finger_down(pinky_tip, pinky_base):
            return "peace"

        # Stop: All fingers extended (open hand with palm facing forward)
        elif is_finger_up(index_tip, index_base) and \
             is_finger_up(middle_tip, middle_base) and \
             is_finger_up(ring_tip, ring_base) and \
             is_finger_up(pinky_tip, pinky_base) and \
             is_finger_up(thumb_tip, thumb_base):
            # Check for palm facing forward by comparing X-axis of the thumb and pinky
            if thumb_tip[0] < pinky_tip[0]:  # Thumb should be left of pinky for palm facing forward
                return "stop"

        # No gesture detected
        return None

    except Exception as e:
        print(f"Error in gesture detection: {e}")
        return None

# Draw landmarks on the frame for debugging
def draw_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

@app.route('/process', methods=['POST'])
def process_frame():
    try:
        # Check for frame in request
        if 'frame' not in request.files:
            return jsonify({"error": "No frame data received"}), 400

        # Read and decode the frame
        frame = request.files['frame'].read()
        nparr = np.frombuffer(frame, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Failed to decode frame"}), 400

        # Convert frame to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = frame.shape
        results = hands.process(rgb_frame)

        # Process landmarks and detect gestures
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Normalize landmarks
                normalized_landmarks = normalize_landmarks(
                    hand_landmarks.landmark, image_width, image_height
                )
                gesture = detect_gesture(normalized_landmarks)

                if gesture and gesture in GESTURE_TO_LANGUAGE:
                    # Draw landmarks for debugging
                    draw_landmarks(frame, results)
                    # Return gesture response
                    return jsonify(GESTURE_TO_LANGUAGE[gesture])

        return jsonify({"error": "No gesture detected"}), 400

    except Exception as e:
        print(f"Error in /process endpoint: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
