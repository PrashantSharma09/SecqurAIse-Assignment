import cv2
import numpy as np
import pandas as pd

# Function to detect colored balls
def detect_balls(frame, lower_color, upper_color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter out small contours
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append((cX, cY))
    return centers

def get_quadrant(x, y, width, height):
    if x < width // 2 and y < height // 2:
        return 1
    elif x >= width // 2 and y < height // 2:
        return 2
    elif x < width // 2 and y >= height // 2:
        return 3
    else:
        return 4

# Initializing video capture
cap = cv2.VideoCapture('input_video.mp4')
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Initializing parameters
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_count = 0
event_log = []


color_ranges = {
    'Red': ([0, 120, 70], [10, 255, 255]),
    'Green': ([36, 25, 25], [70, 255, 255]),
    'Blue': ([94, 80, 2], [126, 255, 255])
}


prev_positions = {color: [] for color in color_ranges.keys()}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    timestamp = frame_count / 20.0  

    for color, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)

        centers = detect_balls(frame, lower, upper)
        for center in centers:
            cv2.circle(frame, center, 10, (0, 255, 255), -1)
            quadrant = get_quadrant(center[0], center[1], frame_width, frame_height)

            for prev_center in prev_positions[color]:
                prev_quadrant = get_quadrant(prev_center[0], prev_center[1], frame_width, frame_height)
                if prev_quadrant != quadrant:
                    event_type = 'Exit' if prev_quadrant != quadrant else 'Entry'
                    event_log.append([timestamp, prev_quadrant, color, 'Exit'])
                    event_log.append([timestamp, quadrant, color, 'Entry'])
                    cv2.putText(frame, f"{event_type} {color}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        prev_positions[color] = centers

    out.write(frame)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Saving event log to CSV
df = pd.DataFrame(event_log, columns=['Time', 'Quadrant', 'Color', 'Type'])
df.to_csv('event_log.csv', index=False)

print("Processing complete. Video saved as 'output_video.avi' and events saved as 'event_log.csv'.")
