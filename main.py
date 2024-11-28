import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture('demoVideo/1.mp4')

mppose=mp.solutions.pose
pose=mppose.Pose()
mpdraw=mp.solutions.drawing_utils
ptime=0
   

while True:
    success , img = cap.read()

    
    
    if not success:  # Exit the loop if the frame is not successfully read
        print(" end of video.")
        cv2.destroyAllWindows()
        break
    scale=0.50

    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result=pose.process(imgRGB)
    print(result.pose_landmarks)
    if result.pose_landmarks:
        mpdraw.draw_landmarks(img,result.pose_landmarks,mppose.POSE_CONNECTIONS)
        for id,lms in enumerate(result.pose_landmarks.landmark):
            #print(id,lms)
            h,w=img.shape[:2]
            cx,cy=int(lms.x*w),int(lms.y*h)
            cv2.circle(img,(cx,cy),10,(0,255,0),cv2.FILLED)
    width=int(img.shape[1]*scale)
    height=int(img.shape[0]*scale)

    ctime=time.time() 
    fps=1/(ctime-ptime)
    ptime=ctime

    
    img_resize=cv2.resize(img,(width,height),interpolation=cv2.INTER_AREA)
    cv2.putText(img_resize,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,5,(0,0,255),5)
    cv2.imshow("img",img_resize)

    if cv2.waitKey(30) & 0xFF==ord("q"):
        cv2.destroyAllWindows()
        break