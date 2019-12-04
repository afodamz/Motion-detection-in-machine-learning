import cv2, time

first_fram = None

vid = cv2.VideoCapture(0) #create a video capture

while True:
    check, frame = vid.read(0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert the frame color to gray

    gray = cv2.GaussianBlur(gray, (21, 21), 0)  #covert the gray scale to gaussian blur

    if first_fram is None:
        first_fram = gray
        continue            #store the first image/frame of the video and continue

    delta_fra = cv2.absdiff(first_fram, gray)   #calculate the difference between the first frame and other frames


    thresh_fra = cv2.threshold(delta_fra, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_fra = cv2.dilate(thresh_fra, None, iterations=0)     #provides a threshold difference value with less
                                                                # value, that will convert the difference value with
                                                                #less than 30 to black,
                                                                #if the difference value is greater than 30, it
                                                                #will convert those pixels to white


    cnts,_ = cv2.findContours(thresh_fra.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  #define the contour area,
                                                                                                # basically add the border

    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue            #Removes noise and shadows, basically it will keep only that part
                                #white, which has area greater than 1000 pixels

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)   #Create a rectangular box around the object in the frame

    cv2.imshow('frame', frame)
    # cv2.imshow('capturing', gray)
    # cv2.imshow('delta', delta_fra)
    # cv2.imshow('thresh', thresh_fra)


    key = cv2.waitKey(1)

    if key == ord('q'):
        break


vid.release()
cv2.destroyAllWindows()