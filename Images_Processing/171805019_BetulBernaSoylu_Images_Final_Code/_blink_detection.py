import cv2
import dlib
from scipy.spatial import distance

# founding distance between eye landmarks
# vertical eye marks (x, y)-coordinates
def openness(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	result = (A+B)/(2.0*C)
	return round(result, 2)

# landmarks numbers can be seen in
# https://medium.datadriveninvestor.com/training-alternative-dlib-shape-predictor-models-using-python-d1d8f8bd9f5c
# We only need landmarks to calculate openness, so we take what we need
def	draw_eye_and_return_landmark_coordinates(start, end):
	coordinates = list()
	for n in range(start, end):
		next_point = n+1
		# we need this control for finishing circle lines
		if next_point == end:
			next_point = start
		x1 = face_landmarks.part(n).x
		y1 = face_landmarks.part(n).y
		x2 = face_landmarks.part(next_point).x
		y2 = face_landmarks.part(next_point).y

		# coordinates for landmarks we need
		coordinates.append((x1, y1))
		# we draw lines around the eye
		cv2.line(frame,(x1,y1),(x2,y2),(77,255,220), 1)
	return coordinates

# taking live video
cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# shape_predictor_68_face_landmarks.dat is from
# https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/

while True:
	_, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = hog_face_detector(gray)
	for face in faces:
		face_landmarks = dlib_facelandmark(gray, face)
		leftEye = draw_eye_and_return_landmark_coordinates(36, 42)
		rightEye = draw_eye_and_return_landmark_coordinates(42, 48)
		left = openness(leftEye)
		right = openness(rightEye)
		# camera reverses video so I changed texts
		text = str()
		if left<0.23 and right<0.23:
			text = "BOTH EYES ARE CLOSED"
		elif left<0.23:
			text = "RIGHT EYE IS CLOSED"
		elif right<0.23:
			text = "LEFT EYE IS CLOSED"
		else:
			text = "AWAKE"

		cv2.putText(frame,text,(30,50), cv2.FONT_HERSHEY_TRIPLEX,1,(255,0,0),2)
		print("Left : {}	Right : {}".format(left,right))
	cv2.imshow("Eye blink check", frame)
	key = cv2.waitKey(1)
	if key == 27:
		break
cap.release()
cv2.destroyAllWindows()