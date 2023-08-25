import cv2
import mediapipe as mp
import math


class FaceMeshAnalyzer:
    def __init__(self, video_path="Videos/2.mp4"):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=4)

        # Drawing specifications
        self.landmark_drawing_spec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
        self.connection_drawing_spec = self.mpDraw.DrawingSpec(thickness=1)

        self.video_path = video_path

        # Capture source
        self.source = self.choose_input_source()

    def choose_input_source(self):
        print("Choose input source:")
        print("1: Live Camera")
        print("2: Predefined Video File")
        choice = input("Enter your choice (1/2): ")

        if choice == '1':
            return 0
        elif choice == '2':
            return self.video_path
        else:
            print("Invalid choice!")
            exit()

    def compute_distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def analyze(self):
        cap = cv2.VideoCapture(self.source)

        while True:
            success, img = cap.read()
            if not success:
                print("Failed to grab frame.")
                break

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.faceMesh.process(imgRGB)

            img_blank = img.copy()
            alpha = 0.5
            img_blend = cv2.addWeighted(img, 1 - alpha, img_blank, alpha, 0)

            if results.multi_face_landmarks:
                for faceLms in results.multi_face_landmarks:
                    self.mpDraw.draw_landmarks(img_blank, faceLms, self.mpFaceMesh.FACEMESH_FACE_OVAL,
                                               self.landmark_drawing_spec, self.connection_drawing_spec)
                    self.mpDraw.draw_landmarks(img_blank, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                               self.landmark_drawing_spec, self.connection_drawing_spec)
                    img_blend = cv2.addWeighted(img, 1 - alpha, img_blank, alpha, 0)

                    # Check for closed eyes
                    left_eye_distance = self.compute_distance(faceLms.landmark[159], faceLms.landmark[145])
                    right_eye_distance = self.compute_distance(faceLms.landmark[386], faceLms.landmark[374])
                    eye_closed_threshold = 0.017
                    if left_eye_distance < eye_closed_threshold or right_eye_distance < eye_closed_threshold:
                        cv2.putText(img_blend, 'Eyes Closed!', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Check for smile
                    smile_threshold = 0.017
                    mouth_distance = self.compute_distance(faceLms.landmark[61], faceLms.landmark[291])
                    if mouth_distance > smile_threshold:
                        cv2.putText(img_blend, 'SMILE!', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(img_blend, 'DANGER!', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Annotate each landmark
                for i, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    cv2.putText(img_blank, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

            cv2.imshow("Output", img_blend)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# Use the class
analyzer = FaceMeshAnalyzer()
analyzer.analyze()
