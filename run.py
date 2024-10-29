import cv2
import numpy as np

# Constants
MIN_AREA = 400
OVERLAP_THRESHOLD = 0.05  # NMS threshold
KERNEL = np.ones((3, 3), np.uint8)  # Morphological kernel

def initialize_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Cannot read video.")
        cap.release()
        return None, None, None
    height, width, _ = frame.shape
    return cap, height, width

def define_regions(height, width):
    roi_start_y = int(height / 2)
    lane_height = height - roi_start_y
    left_lane_area = ((0, 0), (width // 2, lane_height))
    right_lane_area = ((width // 2, 0), (width, lane_height))
    direction_line_y = int(lane_height * 0.4)  # 40% height for line
    return roi_start_y, lane_height, left_lane_area, right_lane_area, direction_line_y

def preprocess_frame(frame, roi_start_y):
    roi = frame[roi_start_y:]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return roi, gray

def get_contours(prev_gray, current_gray):
    frame_diff = cv2.absdiff(prev_gray, current_gray)
    blurred = cv2.medianBlur(frame_diff, 5)
    _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, KERNEL, iterations=1)
    thresh = cv2.erode(thresh, KERNEL, iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def apply_nms(contours):
    boxes, confidences = [], []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MIN_AREA:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            if 0.5 < aspect_ratio < 5.0:  # Aspect ratio filter
                boxes.append([x, y, w, h])
                confidences.append(float(area))  # Use area as a confidence score
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0, nms_threshold=OVERLAP_THRESHOLD)
    return boxes, indices

def update_vehicle_states(boxes, indices, left_lane_area, right_lane_area, direction_line_y, vehicle_positions):
    left_lane_count = 0
    right_lane_count = 0
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            center_x, center_y = x + w // 2, y + h // 2

            # Lane counting
            if left_lane_area[0][0] < center_x < left_lane_area[1][0] and left_lane_area[0][1] < center_y < left_lane_area[1][1]:
                left_lane_count += 1
            elif right_lane_area[0][0] < center_x < right_lane_area[1][0] and right_lane_area[0][1] < center_y < right_lane_area[1][1]:
                right_lane_count += 1

            # Determine crossing status and color
            vehicle_id = (x, y, w, h)
            previous_data = vehicle_positions.get(vehicle_id, {"y": center_y, "crossed": False, "color": (255, 255, 0)})
            if not previous_data["crossed"] and center_y >= direction_line_y:
                if left_lane_area[0][0] < center_x < left_lane_area[1][0]:
                    previous_data["color"] = (0, 0, 255)  # Red for left lane
                elif right_lane_area[0][0] < center_x < right_lane_area[1][0]:
                    previous_data["color"] = (0, 255, 0)  # Green for right lane
                previous_data["crossed"] = True

            vehicle_positions[vehicle_id] = previous_data
            color = previous_data["color"]
            yield (x, y, w, h, color)

    return left_lane_count, right_lane_count

def draw_annotations(roi, boxes_colors, left_lane_area, right_lane_area, direction_line_y, left_lane_count, right_lane_count):
    # Draw bounding boxes with colors
    for (x, y, w, h, color) in boxes_colors:
        cv2.rectangle(roi, (x, y), (x + w, y + h), color, 2)

    # Draw lane areas
    cv2.rectangle(roi, left_lane_area[0], left_lane_area[1], (255, 255, 255), 2)
    cv2.rectangle(roi, right_lane_area[0], right_lane_area[1], (255, 255, 255), 2)

    # Draw the direction line
    cv2.line(roi, (0, direction_line_y), (roi.shape[1], direction_line_y), (0, 255, 255), 2)

    # Display lane counts
    text = f"Left lane: {left_lane_count} | Right lane: {right_lane_count}"
    cv2.putText(roi, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

def main(video_path):
    cap, height, width = initialize_video(video_path)
    if cap is None:
        return

    roi_start_y, lane_height, left_lane_area, right_lane_area, direction_line_y = define_regions(height, width)
    _, prev_frame = cap.read()
    roi, prev_gray = preprocess_frame(prev_frame, roi_start_y)
    vehicle_positions = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the current frame
        roi, gray = preprocess_frame(frame, roi_start_y)
        contours = get_contours(prev_gray, gray)
        boxes, indices = apply_nms(contours)

        # Update vehicle states and get colors
        boxes_colors = list(update_vehicle_states(boxes, indices, left_lane_area, right_lane_area, direction_line_y, vehicle_positions))

        # Count vehicles in each lane
        left_lane_count, right_lane_count = len([1 for box in boxes_colors if box[4] == (0, 0, 255)]), len([1 for box in boxes_colors if box[4] == (0, 255, 0)])

        # Draw everything on the ROI
        draw_annotations(roi, boxes_colors, left_lane_area, right_lane_area, direction_line_y, left_lane_count, right_lane_count)

        # Update the previous frame for the next iteration
        prev_gray = gray
        frame[roi_start_y:height, 0:width] = roi
        cv2.imshow('Vehicle Detection and Color Change', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the main function
main("1952-152220070_small.mp4")
