import cv2
import math
import time
import os
import subprocess
import pyautogui
import pyttsx3
import speech_recognition as sr
import wikipedia
import webbrowser as wb
import pyjokes
from ultralytics import YOLO
from time import sleep
import threading
import numpy as np
from sympy import symbols, sympify, solve, lambdify
import re
from datetime import datetime, timedelta

# Configuration settings
CONFIDENCE_THRESHOLD = 0.60  # Confidence threshold for object detection
OBJECTS_OF_INTEREST = ["person", "chair", "cup", "bottle", "mouse", "cell phone"]  # Your specific objects
DISPLAY_TIME = 3  # Seconds to display each detection frame
CAMERA_WIDTH = 480  # Reduced resolution (was 640)
CAMERA_HEIGHT = 360  # Reduced resolution (was 480)
PROCESS_EVERY_N_FRAMES = 10  # Only process every Nth frame to reduce lag

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

# Path for Brave Browser
brave_path = "C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe %s"

# Global variables for threading
running = False
img = None
frame_buffer = []  # Buffer to store frames for display between processing

# Global detection history for verbal descriptions
detection_history = {
    "last_announced": {},  # Last announced objects
    "last_announcement_time": 0,  # Time of last announcement
    "detection_counts": {},  # Counting objects over time
    "scene_description": ""  # Overall scene description
}

# Enhanced object descriptions for more natural verbal output
object_descriptions = {
    "person": ["someone", "a person", "an individual"],
    "chair": ["a chair", "a seat"],
    "cup": ["a cup", "a mug", "a drinking vessel"],
    "bottle": ["a bottle", "a water bottle", "a container"],
    "mouse": ["a computer mouse", "a mouse"],
    "cell phone": ["a mobile phone", "a cell phone", "a smartphone"],
    # Add more objects and descriptions as needed
}
# ========================
# Math Calculation Functions
# ========================

def calculate_expression(expression):
    """Evaluate a mathematical expression safely"""
    try:
        # Check for potential code execution risk
        if any(keyword in expression for keyword in ['import', 'exec', 'eval', 'os.', 'subprocess', '__']):
            return "I cannot process this expression due to security concerns."

        # For basic arithmetic, we can use eval with safe restrictions
        if all(c in "0123456789+-*/().^ " for c in expression):
            # Replace ^ with ** for power
            expression = expression.replace('^', '**')
            result = eval(expression, {"__builtins__": {}})
            return f"The result of {expression} is {result}"

        # For more complex expressions, try sympy
        x, y, z = symbols('x y z')
        expr = sympify(expression)
        result = expr.evalf()
        return f"The result of {expression} is {result}"

    except Exception as e:
        return f"I couldn't calculate that: {str(e)}"


def solve_equation(equation):
    """Solve an algebraic equation"""
    try:
        # Extract variable and equation parts
        match = re.search(r'for\s+([a-zA-Z])', equation)
        if match:
            var_name = match.group(1)
            equation = re.sub(r'for\s+[a-zA-Z]', '', equation)
        else:
            var_name = 'x'  # Default variable

        # Clean up the equation
        equation = equation.replace("solve", "").replace("equation", "").strip()

        # Handle "equal to" or "=" format
        if "=" in equation:
            left, right = equation.split("=", 1)
            equation = f"{left}-({right})"

        var = symbols(var_name)
        expr = sympify(equation)

        solution = solve(expr, var)
        if solution:
            if len(solution) == 1:
                return f"The solution for {var_name} is {solution[0]}"
            else:
                return f"The solutions for {var_name} are {', '.join(str(sol) for sol in solution)}"
        else:
            return "No solution found for this equation."

    except Exception as e:
        return f"I couldn't solve that equation: {str(e)}"


def perform_calculation(query):
    """Parse user query and perform appropriate calculation"""
    query = query.lower()

    # Check for specific math operations
    if "solve" in query and ("equation" in query or "for" in query):
        return solve_equation(query)

    elif any(word in query for word in ["calculate", "compute", "what is", "result of"]):
        # Extract the mathematical expression
        expression = query.replace("calculate", "").replace("compute", "").replace("what is", "").replace(
            "result of", "").strip()
        return calculate_expression(expression)

    elif any(op in query for op in
             ["+", "-", "*", "/", "^", "squared", "cubed", "square root", "cube root"]):
        # Convert word operations to symbols
        query = query.replace("squared", "^2")
        query = query.replace("cubed", "^3")
        query = query.replace("square root of", "sqrt")
        query = query.replace("cube root of", "cbrt")

        # Extract numbers and operations
        expression = ''.join([c for c in query if c.isdigit() or c in "+-*/^.()"])
        return calculate_expression(expression)

    else:
        return "I'm not sure what calculation you want me to perform. Please be more specific."


# ========================
# Basic Assistant Functions
# ========================

# Speak function
def speak(audio):
    engine.say(audio)
    engine.runAndWait()


# Listen for commands
def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-in')
        print(f"Command: {query}")
        return query.lower()
    except:
        return "None"


# Function to get current time
def time_now():
    current_time = time.strftime("%H:%M:%S")
    speak(f"The current time is {current_time}")


# Function to get today's date
def date_today():
    day = int(time.strftime("%d"))
    month = int(time.strftime("%m"))
    year = int(time.strftime("%Y"))
    speak(f"Today's date is {day} {month} {year}")


# Welcome message
def wishme():
    speak("Welcome back Sir")
    hour = int(time.strftime("%H"))

    if 6 <= hour < 12:
        speak("Good Morning")
    elif 12 <= hour < 18:
        speak("Good Afternoon")
    elif 18 <= hour <= 24:
        speak("Good Evening")
    else:
        speak("Good Night")

    speak("How can I assist you today?")


# Open Brave Browser
def open_brave():
    speak("Opening Brave browser")
    wb.get(brave_path).open("https://www.google.com")


# ========================
# Enhanced Object Detection Functions
# ========================

# Positional descriptions for object location
def get_position_description(x1, y1, x2, y2, img_width, img_height):
    """Get verbal description of object position in frame"""
    # Calculate center point
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Horizontal position
    if center_x < img_width * 0.33:
        h_pos = "on the left side"
    elif center_x < img_width * 0.66:
        h_pos = "in the middle"
    else:
        h_pos = "on the right side"

    # Vertical position
    if center_y < img_height * 0.33:
        v_pos = "at the top"
    elif center_y < img_height * 0.66:
        v_pos = "in the center"
    else:
        v_pos = "at the bottom"

    # Size description based on box area compared to image area
    box_area = (x2 - x1) * (y2 - y1)
    img_area = img_width * img_height
    ratio = box_area / img_area

    if ratio < 0.05:
        size = "small"
    elif ratio < 0.15:
        size = "medium-sized"
    else:
        size = "large"

    return f"a {size} object {h_pos} {v_pos}"


# Function to generate verbal description for detected objects
def generate_verbal_description(detected_objects, img_shape):
    """Generate a natural language description of detected objects"""
    global detection_history

    current_time = time.time()
    img_height, img_width = img_shape[:2]

    # Initialize variables
    new_objects = []
    all_objects = []

    # Update object counts
    for obj in detected_objects:
        obj_class = obj["class"]
        position = obj["position"]  # (x1, y1, x2, y2)
        confidence = obj["confidence"]

        # Track object counts
        if obj_class in detection_history["detection_counts"]:
            detection_history["detection_counts"][obj_class] += 1
        else:
            detection_history["detection_counts"][obj_class] = 1

        # Check if this is a new detection
        is_new = False
        if obj_class not in detection_history["last_announced"]:
            is_new = True
        elif current_time - detection_history[
            "last_announcement_time"] > 10:  # Announce again after 10 seconds
            is_new = True

        # Get position description
        pos_desc = get_position_description(*position, img_width, img_height)

        # Get varied object description for more natural speech
        if obj_class in object_descriptions:
            obj_desc = np.random.choice(object_descriptions[obj_class])
        else:
            obj_desc = f"a {obj_class}"

        # Create description
        description = f"{obj_desc} {pos_desc}"

        if is_new:
            new_objects.append(description)

        all_objects.append({
            "description": description,
            "class": obj_class,
            "position": position,
            "confidence": confidence
        })

    # Generate scene description
    if not all_objects:
        scene_description = "I don't see any objects of interest at the moment."
    else:
        # Update last announced
        for obj in all_objects:
            detection_history["last_announced"][obj["class"]] = obj["position"]

        detection_history["last_announcement_time"] = current_time

        # Create scene description
        if len(all_objects) == 1:
            scene_description = f"I can see {all_objects[0]['description']}."
        else:
            obj_descriptions = [obj["description"] for obj in all_objects]
            scene_description = f"I can see {', '.join(obj_descriptions[:-1])} and {obj_descriptions[-1]}."

    # Create announcement for new objects
    if new_objects:
        if len(new_objects) == 1:
            announcement = f"I just noticed {new_objects[0]}."
        else:
            announcement = f"I just noticed {', '.join(new_objects[:-1])} and {new_objects[-1]}."
    else:
        announcement = ""

    # Update the history
    detection_history["scene_description"] = scene_description

    return {
        "scene_description": scene_description,
        "announcement": announcement
    }


# Function to speak object descriptions (to be called in a non-blocking way)
def speak_object_description(detected_objects, img_shape, force_speak=False):
    """Generate and speak descriptions of detected objects"""
    descriptions = generate_verbal_description(detected_objects, img_shape)

    # Decide what to announce
    if force_speak:
        # If forced, announce the full scene description
        speak_text = descriptions["scene_description"]
    elif descriptions["announcement"]:
        # If there are new objects, announce them
        speak_text = descriptions["announcement"]
    else:
        # Nothing new to announce
        return

    # Use the existing speak function in a non-blocking way
    speak_thread = threading.Thread(target=speak, args=(speak_text,))
    speak_thread.daemon = True
    speak_thread.start()


# Camera capture thread function - optimized for smoother display
def camera_capture(cap):
    global img, running, frame_buffer
    frame_count = 0

    while running:
        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue

        # Store every frame for display, but only some for processing
        img = frame.copy()
        frame_count += 1

        # Keep a reasonable buffer size
        if len(frame_buffer) > 5:
            frame_buffer.pop(0)

        # Add current frame to buffer
        if frame_count % 3 == 0:  # Add every 3rd frame to buffer for smoother playback
            frame_buffer.append(frame.copy())

        # Prevent thread from consuming too much CPU
        time.sleep(0.01)


# Voice command monitoring thread
def voice_monitor():
    global running
    while running:
        command = takeCommand()
        if "stop detection" in command or "stop" in command:
            speak("Stopping object detection.")
            running = False
        elif "what do you see" in command:
            # Trigger a verbal description of the current scene
            if 'detected_objects' in globals() and detected_objects:
                speak_object_description(detected_objects, img.shape, force_speak=True)
            else:
                speak("I don't see any objects of interest at the moment.")
        time.sleep(0.5)  # Check commands every half second


# YOLO Object Detection with verbal descriptions
def object_detection():
    global running, img, frame_buffer, detected_objects

    speak("Starting object detection... Say 'stop detection' to exit.")

    try:
        # Initialize camera with modest resolution for better performance
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # Try to set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering

        if not cap.isOpened():
            speak("Failed to open camera. Please check your connection.")
            return

        # Initialize model with the smallest available YOLO model
        model = YOLO("yolo-Weights/yolov8n.pt")

        classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                      "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                      "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                      "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                      "sports ball",
                      "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                      "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                      "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                      "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop",
                      "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                      "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
                      "toothbrush"]

        # Start threads
        running = True
        img = None
        frame_buffer = []
        detected_objects = []  # Initialize global detected_objects

        # Start camera capture in a separate thread
        capture_thread = threading.Thread(target=camera_capture, args=(cap,))
        capture_thread.daemon = True
        capture_thread.start()

        # Start voice monitoring in a separate thread
        voice_thread = threading.Thread(target=voice_monitor)
        voice_thread.daemon = True
        voice_thread.start()

        # Variables for frame processing
        frame_count = 0
        last_process_time = time.time()
        last_verbal_time = time.time()  # Track when we last gave a verbal update
        processed_img = None
        detections_text = "Starting detection..."

        # Window setup (allows resizing)
        cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Object Detection", CAMERA_WIDTH, CAMERA_HEIGHT)

        while running:
            current_time = time.time()

            # Handle case where no frames are available yet
            if img is None and not frame_buffer:
                time.sleep(0.1)
                continue

            # Display logic - always show something
            display_img = None

            # Time to process a new frame?
            process_now = (current_time - last_process_time >= DISPLAY_TIME) and (
                    processed_img is not None or frame_count % PROCESS_EVERY_N_FRAMES == 0)

            if process_now:
                # Use the latest frame for processing
                if img is not None:
                    current_img = img.copy()
                elif frame_buffer:
                    current_img = frame_buffer[-1].copy()
                else:
                    continue

                # Clear previous detections
                detections_found = []
                detected_objects = []  # Clear the detected objects list

                # Run model on current frame - updated with current YOLO parameters
                results = model(current_img, imgsz=320)  # Using imgsz instead of size

                # Process detections
                for r in results:
                    for box in r.boxes:
                        confidence = box.conf[0]

                        # Skip low confidence detections
                        if confidence < CONFIDENCE_THRESHOLD:
                            continue

                        # Get class information
                        class_id = int(box.cls[0])
                        class_name = classNames[class_id]

                        # Skip objects we're not interested in
                        if class_name not in OBJECTS_OF_INTEREST:
                            continue

                        # Add to detected objects list
                        confidence_percentage = math.ceil(confidence * 100)
                        detections_found.append(f"{class_name}: {confidence_percentage}%")

                        # Draw bounding box and label
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(current_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        label = f"{class_name} {confidence_percentage}%"
                        cv2.putText(current_img, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Store detected object details for verbal description
                        detected_objects.append({
                            "class": class_name,
                            "confidence": confidence_percentage,
                            "position": (x1, y1, x2, y2)
                        })

                # Generate verbal descriptions if objects were detected
                if detected_objects:
                    # Check if it's time to provide a verbal update (every 5 seconds)
                    if current_time - last_verbal_time >= 5:
                        # Call in non-blocking way to avoid freezing the UI
                        threading.Thread(
                            target=speak_object_description,
                            args=(detected_objects, current_img.shape),
                            daemon=True
                        ).start()
                        last_verbal_time = current_time

                # Update display information
                if detections_found:
                    detections_text = "Detected: " + ", ".join(detections_found)
                else:
                    detections_text = "No objects of interest detected"

                # Add timestamp and detection summary to the frame
                cv2.putText(current_img, f"Time: {time.strftime('%H:%M:%S')}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Add a semi-transparent overlay at the bottom for detection text
                overlay = current_img.copy()
                cv2.rectangle(overlay, (0, current_img.shape[0] - 60),
                              (current_img.shape[1], current_img.shape[0]), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, current_img, 0.3, 0, current_img)

                # Add detection text on the overlay
                y_pos = current_img.shape[0] - 30
                cv2.putText(current_img, detections_text[:60], (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # Update the processed image
                processed_img = current_img
                last_process_time = current_time

                # Print to console for debugging
                print(f"Detection update: {detections_text}")

                # Set the display image to the processed result
                display_img = processed_img
            else:
                # Between processing intervals, display the latest raw frame with overlay text
                if frame_buffer:
                    display_img = frame_buffer[-1].copy()
                elif img is not None:
                    display_img = img.copy()
                else:
                    continue

                # Add prior detection results text
                if processed_img is not None:
                    # Add timestamp
                    cv2.putText(display_img, f"Time: {time.strftime('%H:%M:%S')}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                    # Add overlay
                    overlay = display_img.copy()
                    cv2.rectangle(overlay, (0, display_img.shape[0] - 60),
                                  (display_img.shape[1], display_img.shape[0]), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.7, display_img, 0.3, 0, display_img)

                    # Add last detection text
                    y_pos = display_img.shape[0] - 30
                    cv2.putText(display_img, detections_text[:60], (10, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Always show something if available
            if display_img is not None:
                cv2.imshow("Object Detection", display_img)

            # Increment frame counter
            frame_count += 1

            # Manual exit with 'q' key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                running = False

            # Slow down the display loop slightly
            time.sleep(0.01)

    except Exception as e:
        print(f"Error in object detection: {e}")
        speak("An error occurred during object detection.")
    finally:
        # Clean up resources
        running = False
        if 'capture_thread' in locals():
            capture_thread.join(timeout=1.0)
        if 'voice_thread' in locals():
            voice_thread.join(timeout=1.0)
        if 'cap' in locals() and cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        speak("Object detection stopped.")


# ========================
# New Feature: Reminders
# ========================

# Dictionary to store reminders
reminders = []


def add_reminder():
    """Add a new reminder"""
    speak("What would you like me to remind you about?")
    reminder_text = takeCommand().lower()

    if reminder_text == "none":
        speak("I couldn't understand what you want to be reminded about.")
        return

    speak("When should I remind you? Please specify time in hours and minutes.")
    time_input = takeCommand().lower()

    try:
        # Parse the time input
        # This is a simple implementation that handles formats like "10 30" or "10:30" or "10 hours 30 minutes"
        time_pattern = re.compile(r'(\d+)(?:\s*:|hours|\s+)?\s*(\d*)')
        match = time_pattern.search(time_input)

        if match:
            hours = int(match.group(1))
            minutes = int(match.group(2)) if match.group(2) else 0

            # Create a reminder time
            now = datetime.now()
            reminder_time = now.replace(hour=hours, minute=minutes, second=0)

            # If the time has already passed today, set it for tomorrow
            if reminder_time < now:
                reminder_time += timedelta(days=1)

            # Add to reminders list
            reminders.append({
                "text": reminder_text,
                "time": reminder_time,
                "active": True
            })

            # Format time for verbal confirmation
            time_str = reminder_time.strftime("%I:%M %p")
            speak(f"Reminder set for {time_str}: {reminder_text}")

            # Start the reminder checker thread if it's not already running
            if not any(t.name == "reminder_checker" for t in threading.enumerate()):
                reminder_thread = threading.Thread(target=check_reminders, name="reminder_checker")
                reminder_thread.daemon = True
                reminder_thread.start()

        else:
            speak("I couldn't understand the time format. Please try again.")
    except Exception as e:
        speak(f"I had trouble setting that reminder: {str(e)}")


def list_reminders():
    """List all active reminders"""
    if not reminders:
        speak("You don't have any reminders set.")
        return

    speak("Here are your current reminders:")
    for i, reminder in enumerate(reminders):
        if reminder["active"]:
            time_str = reminder["time"].strftime("%I:%M %p")
            speak(f"Reminder {i + 1}: {reminder['text']} at {time_str}")


def delete_reminder():
    """Delete a specific reminder"""
    if not reminders:
        speak("You don't have any reminders to delete.")
        return

    list_reminders()
    speak("Which reminder would you like to delete? Please say the reminder number.")

    choice = takeCommand().lower()
    try:
        # Extract number from response
        number_pattern = re.compile(r'(\d+)')
        match = number_pattern.search(choice)

        if match:
            index = int(match.group(1)) - 1
            if 0 <= index < len(reminders):
                deleted_reminder = reminders[index]
                reminders[index]["active"] = False  # Mark as inactive
                speak(f"Deleted reminder: {deleted_reminder['text']}")
            else:
                speak("Invalid reminder number. Please try again.")
        else:
            speak("I couldn't understand which reminder to delete.")
    except Exception as e:
        speak(f"Error deleting reminder: {str(e)}")


def check_reminders():
    """Background thread to check and trigger reminders"""
    while True:
        current_time = datetime.now()

        for reminder in reminders:
            if reminder["active"]:
                reminder_time = reminder["time"]
                # Check if reminder time is within the last minute
                time_diff = (current_time - reminder_time).total_seconds()
                if 0 <= time_diff < 60:  # Within the last minute
                    speak(f"Reminder: {reminder['text']}")
                    reminder["active"] = False  # Deactivate after triggering

        # Check every 30 seconds
        time.sleep(30)


# ========================
# Main Program Loop
# ========================

if __name__ == "__main__":
    wishme()

    while True:
        query = takeCommand().lower()

        # Basic commands
        if "time" in query:
            time_now()

        elif "date" in query:
            date_today()

        elif "wikipedia" in query:
            speak("Searching Wikipedia...")
            query = query.replace("wikipedia", "")
            results = wikipedia.summary(query, sentences=2)
            speak("According to Wikipedia")
            speak(results)

        elif "open browser" in query or "open brave" in query:
            open_brave()

        elif "joke" in query:
            joke = pyjokes.get_joke()
            speak(joke)

        elif "screenshot" in query:
            img = pyautogui.screenshot()
            img.save(f"screenshot_{int(time.time())}.png")
            speak("Screenshot taken and saved")

        # Calculator functions
        elif any(word in query for word in ["calculate", "compute", "math", "equation", "solve"]):
            result = perform_calculation(query)
            speak(result)

        # Camera and object detection
        elif "detect objects" in query or "what can you see" in query or "use camera" in query:
            object_detection()

        # Reminder functions
        elif "set reminder" in query or "remind me" in query:
            add_reminder()

        elif "list reminders" in query or "show reminders" in query:
            list_reminders()

        elif "delete reminder" in query:
            delete_reminder()

        # Exit command
        elif "exit" in query or "quit" in query or "goodbye" in query:
            speak("Goodbye! Have a great day.")
            break

        # If no command matched
        elif query != "none":
            speak("I didn't understand that command. Please try again.")


        time.sleep(0.2)
