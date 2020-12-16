import cv2
 
vidcap = cv2.VideoCapture('/mnt/test.mp4')
 
count = 0
 
while(vidcap.isOpened()):
   
    ret, image = vidcap.read()
 
    cv2.imwrite("/mnt/ITRC/flownet2-pytorch/dataset/test/frame%04d.jpg" % count, image)
 
    print('Saved frame%d.jpg' % count)
    count += 1
 
vidcap.release()

