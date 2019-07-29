import numpy as np
import cv2
import copy
import math

#Feeding the required video and image files to be run on the code
c=cv2.VideoCapture('multipleTags.mp4')
image1=cv2.imread('Lena.png')

#Dividing the obtained AR Tag in the image plane to 8*8 matrix
def TagMatrix(img):
    dimension_tag = img.shape		# Calculate the shape of the image
    height_img = dimension_tag[0]
    width_img = dimension_tag[1]
    bitheight = int((height_img/8))
    bitwidth = int(width_img/8)
    countblack = 0
    countwhite = 0
    a=0
    b=0
    ar_tag = np.empty((8,8))		#Initialising the 8X8 matrix
    for i in range(0,height_img,bitheight):
        b=0
        for j in range(0,width_img,bitwidth):
            countblack=0
            countwhite=0
            for x in range(0,bitheight-1):
                for y in range(0,bitwidth-1):
                    if(img[i+x][j+y]==0):
                        countblack = countblack + 1
                    else:
                        countwhite = countwhite + 1
                        
            if(countwhite >= countblack):	# Checking whether that block has more white or black pixel and corresponding assigning it in the tag matrix
                ar_tag[a][b]=1
            else:
                ar_tag[a][b]=0
            #print(artag)
            b=b+1
        a=a+1
    return ar_tag

#Comparing the inner 4x4 grid to check for the orientation of the tag
def TagAngle(artag):
    # Checking the location of white block in the inner 4X4 matrix of the AR tag to detect the orientation of the tag in camera frame
    if(artag[2][2] == 0 and artag[2][5] == 0 and artag[5][2] == 0 and artag[5][5] == 1):
        twist = 0
    elif(artag[2][2] == 1 and artag[2][5] == 0 and artag[5][2] == 0 and artag[5][5] == 0):
        twist = 180
    elif(artag[2][2] == 0 and artag[2][5] == 1 and artag[5][2] == 0 and artag[5][5] == 0):
        twist = 90
    elif(artag[2][2] == 0 and artag[2][5] == 0 and artag[5][2] == 1 and artag[5][5] == 0):
        twist = -90
    else:
        twist = None
        
    if (twist == None):
        return twist, False
    else:
        return twist, True
    

#Finding the Binary value of the  tag using the inner 2x2 matrix 
def TagId(image):

    tag_matrix = TagMatrix(image)

    angle_value , flag = TagAngle(tag_matrix)
     
    if (flag == False):		# Checking the tag is detected or not.
        
        return flag , angle_value , None
        
    if(flag == True):	 #Based on the orientation detected the binary value of the tag is calculated
        if (angle_value == 0):
            Id = tag_matrix[3][3]*1 +tag_matrix[4][3]*8 +tag_matrix[4][4]*4 + tag_matrix[3][4]*2
        elif(angle_value == 90):
            Id = tag_matrix[3][3]*2 + tag_matrix[3][4]*4 + tag_matrix[4][4]*8 +tag_matrix[4][3]*1
        elif(angle_value == 180):
            Id = tag_matrix[3][3]*4 + tag_matrix[4][3]*2 + tag_matrix[4][4] + tag_matrix[3][4]*8
        elif(angle_value == -90):
            Id= tag_matrix[3][3]*8 + tag_matrix[3][4] + tag_matrix[4][4]*2 +tag_matrix[4][3]*4
        return flag, angle_value, Id

#Finding the Homography of the AR tag from World Coordinate frame to Image Coordinate frame
def find_homography(img1, img2):
    ind = 0
    A_matrix = np.empty((8, 9))
    
    for pixel in range(0, len(img1)):
        
        x_1 = img1[pixel][0]	#Extracting pixel of world frame
        y_1 = img1[pixel][1]

        x_2 = img2[pixel][0]	#Extracting pixel of camera frame
        y_2 = img2[pixel][1]

        A_matrix[ind] = np.array([x_1, y_1, 1, 0, 0, 0, -x_2*x_1, -x_2*y_1, -x_2])
        A_matrix[ind + 1] = np.array([0, 0, 0, x_1, y_1, 1, -y_2*x_1, -y_2*y_1, -y_2])

        ind = ind + 2
    
    U, s, V = np.linalg.svd(A_matrix, full_matrices=True)
    V = (copy.deepcopy(V)) / (copy.deepcopy(V[8][8]))	#Extracting value of last Eigen vector and dividing the z component with the x and y
    H = V[8,:].reshape(3, 3)
    return H


#Extracting frames from the video to blur followed by converting it to grayscale and then thresholding the pixels
while (True):
    ret,image=c.read()
    #image1=cv2.imread('Lena.png')
    blurred = cv2.GaussianBlur(image,(7,7),0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

#Finding the contours present in the video
    _,contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    list_for_corners=[]

#From all the contours, extracting the required contours by comparing the pixel values of neighbouring pixels
    for i in contours:
        corner_four=[]		#To check 4 corners are there
        if cv2.contourArea(i) > 1000:
            epsilon = 0.1*cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i, epsilon, True)
            for j in approx:
                count = 0 
                topleft, topright, bottomleft, bottomright = 0, 0, 0, 0
                if j[0][0] < (np.shape(threshold)[1]-20) and j[0][1] < (np.shape(threshold)[0]-20):
                    x=j[0][1]
                    y=j[0][0]

#Condition to extract AR Tag corners and the index of the corners to be fed into 'corner_four' list
                    if threshold[x-10][y-10]==255:
                        count=count+1
                        topleft=1
                    if threshold[x+10][y+10]==255:
                        count=count+1
                        bottomright=1
                    if threshold[x-10][y+10]==255:
                        count=count+1
                        topright=1
                    if threshold[x+10][y-10]==255:
                        count=count+1
                        bottomleft=1
                    if count == 3:
                        if topleft == 1 and topright == 1 and bottomleft == 1:
                            corner='TOPLEFT'
                        elif topleft == 1 and topright == 1 and bottomright == 1:
                            corner='TOPRIGHT'
                        elif topleft == 1 and bottomright == 1 and bottomleft == 1:
                            corner='BOTTOMLEFT'
                        elif bottomright== 1 and topright == 1 and bottomleft == 1:
                            corner='BOTTOMRIGHT'
                        cv2.drawContours(image, approx, -1, (0, 0, 255), 3)
                        corner_four.append([y,x,corner])
                        

            if len(corner_four)==4:
                list_for_corners.append(corner_four)
    if list_for_corners != []:		#if listforcorners is not empty then go forward
        for i in range(0,len(list_for_corners)):
            corner_position = [0,0,0,0]	#to put x and y corner values in a list
            for value in list_for_corners[i]:
                if value[-1] == 'TOPLEFT':
                    corner_position[0] = value[0:2]
                elif value[-1] == 'TOPRIGHT':
                    corner_position[1] = value[0:2]
                elif value[-1] == 'BOTTOMLEFT':
                    corner_position[2] = value[0:2]
                elif value[-1] == 'BOTTOMRIGHT':
                    corner_position[3] = value[0:2]
                    
            if 0 not in corner_position:  
                H = find_homography(corner_position, [[0,0],[199,0],[0,199],[199,199]])
                H_inv=np.linalg.inv(H)
                im_out=np.zeros((200,200))
                for a in range(0,200):
                    for b in range(0,200):
                        x, y, z = np.matmul(H_inv,[a,b,1])
                        if (int(y/z) < 1080 and int(y/z) > 0) and (int(x/z) < 1920 and int(x/z) > 0):
                            im_out[a][b] = threshold[int(y/z)][int(x/z)]
                l, angle, identity = TagId(im_out)
                
                if l:
                    if angle == 0:
                        corner_actual = corner_position
                    elif angle == 90:
                        corner_actual = [corner_position[2], corner_position[0], corner_position[3], corner_position[1]]
                    elif angle == -90:
                        corner_actual = [corner_position[1], corner_position[3], corner_position[0], corner_position[2]]
                    elif angle == 180:
                        corner_actual = [corner_position[3], corner_position[2], corner_position[1], corner_position[0]]
                    
                    image1 = cv2.resize(image1, (90, 90))
                    H_new= find_homography(corner_actual, [[0,0],[0,89],[89,0],[89,89]])
                    H_inv = np.linalg.inv(H_new)  
            
                    for j in range(0,90):
                        for k in range(0,90):
                            x_lena, y_lena, z_lena = np.matmul(H_inv,[j,k,1])
                            if (int(y_lena/z_lena) < np.shape(threshold)[0] and int(y_lena/z_lena) > 0) and (int(x_lena/z_lena) < np.shape(threshold)[1] and int(x_lena/z_lena) > 0):
                                image[int(y_lena/z_lena)][int(x_lena/z_lena)] = image1[j][k]
                              
                    
            corner_position = [] 

#Displaying the image output in a window named "Display"
    cv2.imshow("DISPLAY1", image)


    k = cv2.waitKey(1)
    if k == 27:
        break         # wait for ESC key to exit
c.release
cv2.destroyAllWindows()
