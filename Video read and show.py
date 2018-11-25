import cv2
import numpy as np
from timeit import default_timer as timer

#Definiciones de escala y gráfica
z_x_axis_dist = 1*np.pi
z_center_offset = 0-0j
w_x_axis_dist = 5
w_center_offset = 0-0j

trans_lineal_z = lambda z: (z_x_axis_dist/width) * z + z_center_offset - complex( z_x_axis_dist/2 , (z_x_axis_dist*hight)/(2*width) )
trans_lineal_w = lambda w: (width/w_x_axis_dist) * ( w + w_center_offset + complex( w_x_axis_dist/2 , (w_x_axis_dist*hight) / (2*width) ) )

out = cv2.VideoWriter('mapped_EULER.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, ((740)*2,740))

cap = cv2.VideoCapture('rejilla.mp4')

if (cap.isOpened()== False): 
  print("Error opening video stream or file")

ret, frame = cap.read()
img = frame[40:1040, 460:1460]
resized_image = cv2.resize(img, (740, 740))
img = resized_image

hight, width, channels = img.shape
n_of_bytes = img.size
print("Imagen de ", width," x ", hight," pixeles, con ", channels," canales. ", n_of_bytes,"Bytes en total.")

w_plane = np.ndarray((hight,width), np.complex64)
mapX = np.ndarray((hight,width), np.float32)
mapY = np.ndarray((hight,width), np.float32)
#    w = complex( np.sin(x_trans)*np.cosh(y_trans) , np.cos(x_trans)*np.sinh(y_trans))


t = timer()
for x in range(width):
  for y in range(hight):
    z = complex(x,y)

    z_trans = trans_lineal_z(z)
    x_trans = z_trans.real
    y_trans = z_trans.imag
    #---------------------------------------------------------------------------
    #w = z_trans.conjugate()
    #w = z_trans**2
    w = complex( np.sin(x_trans)*np.cosh(y_trans), np.cos(x_trans)*np.sinh(y_trans))


    #---------------------------------------------------------------------------
    w = trans_lineal_w(w)

    real_w = np.around(w.real)
    real_w = real_w.astype(np.int32)
    imag_w = np.around(w.imag)
    imag_w = imag_w.astype(np.int32)
    if real_w>=0 and real_w<=(width-1) and imag_w>=0 and imag_w<=(hight-1):
      w_plane[imag_w,real_w] = z
      mapX[imag_w,real_w] = z.real
      mapY[imag_w,real_w] = z.imag

print("Tiempo en construir la matriz de reordenamiento %.fms" %((timer()-t)*1000))

Map1, Map2 = cv2.convertMaps(mapX, mapY, cv2.CV_16SC2)
 

while(cap.isOpened()):
  t1 = timer()
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
 
    # Display the resulting frame
    #roi = frame[y1:y2, x1:x2]
    img = frame[40:1040, 460:1460]
    resized_image = cv2.resize(img, (740, 740))
    img = resized_image
    
    #bckgrd_white = 160
    #img[0,0] = (bckgrd_white,bckgrd_white,bckgrd_white)

    img_remmaped = cv2.remap(img,Map1,Map2,cv2.INTER_LINEAR)

     # Write some Text-----------------------------------------------------------------------
    font                   = cv2.FONT_HERSHEY_TRIPLEX
    bottomLeftCornerOfText = (width-30,hight-5)
    fontScale              = 1
    fontColor              = (0,0,0)
    lineType               = 1

    cv2.putText(img,'z', 
          bottomLeftCornerOfText, 
          font, 
          fontScale,
          fontColor,
          lineType)
    cv2.putText(img_remmaped,'w', 
          bottomLeftCornerOfText, 
          font, 
          fontScale,
          fontColor,
          lineType) 

    lineColor              = (120,120,120)
    thickness = 1

    center_x = np.around(width/2)
    center_y = np.around(hight/2)
    center_x = center_x.astype(np.int32)
    center_y = center_y.astype(np.int32)

    cv2.line(img,(center_x,center_y),(center_x,10),lineColor,thickness)
    cv2.line(img,(center_x,center_y),(center_x,hight-10),lineColor,thickness)
    cv2.line(img,(center_x,center_y),(10,center_y),lineColor,thickness)
    cv2.line(img,(center_x,center_y),(width-10,center_y),lineColor,thickness)

    cv2.line(img_remmaped,(center_x,center_y),(center_x,10),lineColor,thickness)
    cv2.line(img_remmaped,(center_x,center_y),(center_x,hight-10),lineColor,thickness)
    cv2.line(img_remmaped,(center_x,center_y),(10,center_y),lineColor,thickness)
    cv2.line(img_remmaped,(center_x,center_y),(width-10,center_y),lineColor,thickness)
    #--------------------------------------------------------------------------------------
      
    combinadas = np.concatenate((img, img_remmaped), axis=1)
    
    out.write(combinadas)
    resized_combinadas = cv2.resize(combinadas, (840, 420))
    cv2.imshow("Mapeo funcion compleja", resized_combinadas)

    t2 = timer()
    print(" Tiempo total: %.fms." %((t2-t1)*1000))
  
  if cv2.waitKey(1) == 27:
    print("Pressed ESC")
    cap.release()
    out.release()
    print("Finished")
    cv2.destroyAllWindows()
    break
 

# When everything done, release the video capture object
print("Finished")
cap.release()
out.release()
# Closes all the frames
cv2.destroyAllWindows()