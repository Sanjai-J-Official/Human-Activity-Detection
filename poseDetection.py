import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture('demoVideo/1.mp4')

class Posedetector():
    def __init__(self,mode=False,Upbody=False,smoth=True,detecctConf=0.5,trankconf=0.5):
        self.mode=mode 
        self.Upbody=Upbody
        self.detecctConf=detecctConf
        self.trankconf=trankconf

        self.mppose=mp.solutions.pose
        self.pose=self.mppose.Pose()
        self.mpdraw=mp.solutions.drawing_utils

    def findpose(self,img,draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.result=self.pose.process(imgRGB)
        
        if self.result.pose_landmarks:
            
            if draw:
                self.mpdraw.draw_landmarks(img,self.result.pose_landmarks,self.mppose.POSE_CONNECTIONS)
                for id,lms in enumerate(self.result.pose_landmarks.landmark):
                    #print(id,lms)
                    h,w=img.shape[:2]
                    cx,cy=int(lms.x*w),int(lms.y*h)
                    cv2.circle(img,(cx,cy),10,(0,255,0),cv2.FILLED)
        return img
    def getPosition(self,img,posistion,draw=True):
        list_pos=[]
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.result=self.pose.process(imgRGB)
        
        if self.result.pose_landmarks:
            
            
            for id,lms in enumerate(self.result.pose_landmarks.landmark):
                
                h,w=img.shape[:2]
                if id==posistion:
                    cx,cy=int(lms.x*w),int(lms.y*h)
                    list_pos.append([cx,cy])
            if draw:
                self.mpdraw.draw_landmarks(img,self.result.pose_landmarks,self.mppose.POSE_CONNECTIONS)
        return list_pos



detector=Posedetector()  
def main():
    ptime=0
    while True:
        success , img = cap.read()
     
        if not success:  # Exit the loop if the frame is not successfully read
            print(" end of video.")
            cv2.destroyAllWindows()
            break
        img=detector.findpose(img,draw=True)
        posistion=detector.getPosition(img,10,draw=False)
        print(posistion)
        scale=0.50

         
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

if __name__=="__main__":
    main()