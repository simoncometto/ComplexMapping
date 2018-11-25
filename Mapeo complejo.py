import requests as req
import cv2
import numpy as np
from timeit import default_timer as timer


#Definiciones de escala y gráfica
z_x_axis_dist = 5   #np.pi
z_center_offset = 0-0j
w_x_axis_dist = 5
w_center_offset = 0-0j

trans_lineal_z = lambda z: (z_x_axis_dist/width) * z + z_center_offset - complex( z_x_axis_dist/2 , (z_x_axis_dist*hight)/(2*width) )
trans_lineal_w = lambda w: (width/w_x_axis_dist) * ( w + w_center_offset + complex( w_x_axis_dist/2 , (w_x_axis_dist*hight) / (2*width) ) )

url = "http://192.168.42.129:8080/shot.jpg"
#url = "https://images.fineartamerica.com/images-medium-large/leonhard-euler-swiss-mathematician-science-source.jpg"
#url = "https://is1-ssl.mzstatic.com/image/thumb/Purple118/v4/80/6e/84/806e8432-c658-7a77-a7b8-21489111828d/source/256x256bb.jpg"
#url = "https://www.abc.es/Media/201509/01/historia-del-logotipo-de-Google--644x362.jpg"
#url = "https://http2.mlstatic.com/hojas-de-vitela-de-diseno-clearprint-1000h-con-cuadricula-im-D_NQ_NP_864242-MLC27099213383_032018-F.jpg"
#url = "http://shopforclipart.com/images/color-circle-cliparts/25.jpg"
#url = "https://steamuserimages-a.akamaihd.net/ugc/850478710041488210/880456DD7BA9A644DC9C10F8120D266481512FA1/"
#url = "https://i.stack.imgur.com/IgSmJ.png"
#url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/Conformal_grid_before_M%C3%B6bius_transformation.svg/1200px-Conformal_grid_before_M%C3%B6bius_transformation.svg.png"

r = req.get(url)
imagen_array = np.array(bytearray(r.content), dtype=np.uint8)
img = cv2.imdecode(imagen_array, -1)

#resized_image = cv2.resize(img, (720, 720))
#img = resized_image
#bimg = img[0:900, 200:1400]
resized_image = cv2.resize(img, (740, 740))
img = resized_image

hight, width, channels = img.shape
n_of_bytes = img.size
print("Imagen de ", width," x ", hight," pixeles, con ", channels," canales. ", n_of_bytes,"Bytes en total.")

w_plane = np.ndarray((hight,width), np.complex64)
mapX = np.ndarray((hight,width), np.float32)
mapY = np.ndarray((hight,width), np.float32)

t = timer()
for x in range(width):
	for y in range(hight):
		z = complex(x,y)

		z_trans = trans_lineal_z(z)
		x_trans = z_trans.real
		y_trans = z_trans.imag
		#---------------------------------------------------------------------------
		#w = z_trans.conjugate()
		w = z_trans**2
		#w = complex( np.exp(x_trans)*np.cos(y_trans) , np.exp(x_trans)*np.sin(y_trans))
		#w = complex( np.sin(x_trans)*np.cosh(y_trans) , np.cos(x_trans)*np.sinh(y_trans))

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

#-----------------------------------------------------------------------------------------------------
while True:
	t1=timer()
	r = req.get(url)
	t3 = timer()
	imagen_array = np.array(bytearray(r.content), dtype=np.uint8)
	img = cv2.imdecode(imagen_array, -1)

	#resized_image = cv2.resize(img, (720, 720))
	#img = resized_image

	#img[0,0] = (255,255,255)

	img_remmaped = cv2.remap(img,Map1,Map2,cv2.INTER_LINEAR)

	lineColor = (120,120,120)
	thickness = 1

	center_x = np.around(width/2)
	center_y = np.around(hight/2)
	center_x = center_x.astype(np.int32)
	center_y = center_y.astype(np.int32)

	#cv2.line(img,(center_x,center_y),(center_x,10),lineColor,thickness)
	#cv2.line(img,(center_x,center_y),(center_x,hight-10),lineColor,thickness)
	#cv2.line(img,(center_x,center_y),(10,center_y),lineColor,thickness)
	#cv2.line(img,(center_x,center_y),(width-10,center_y),lineColor,thickness)

	#cv2.line(img_remmaped,(center_x,center_y),(center_x,10),lineColor,thickness)
	#cv2.line(img_remmaped,(center_x,center_y),(center_x,hight-10),lineColor,thickness)
	#cv2.line(img_remmaped,(center_x,center_y),(10,center_y),lineColor,thickness)
	#cv2.line(img_remmaped,(center_x,center_y),(width-10,center_y),lineColor,thickness)
	#--------------------------------------------------------------------------------------
	
	#combinadas = np.concatenate((img, img_remmaped), axis=1)
	combinadas = img_remmaped
	cv2.imshow("Mapeo funcion compleja", combinadas)

	t2 = timer()
	print("Tiempo img request: %.fms" %((t3-t1)*1000), " Tiempo total: %.fms." %((t2-t1)*1000))
	if cv2.waitKey(1) == 27:
		print("Saliendo del programa")
		break

# When everything done, release the video capture object
print("Se perdió la conexión")
# Closes all the frames
cv2.destroyAllWindows()

