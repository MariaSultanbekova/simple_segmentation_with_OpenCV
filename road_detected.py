import cv2 as cv
import numpy as np

cap = cv.VideoCapture(r'C:\Users\Midor\PycharmProjects\pythonProject\venv\videos\road.mp4')

img_size = [200, 360]

while (cv.waitKey(1) != 27):
    ret, frame = cap.read()
    if ret == False:
        print('END')
        break

    resized = cv.resize(frame, (img_size[1], img_size[0]))
    cv.imshow('frame', resized)

    # Making mask for lines on road

    r_channel = resized[:, :, 2]
    binary = np.zeros_like(r_channel)
    binary[(r_channel > 150)] = 1

    hsv = cv.cvtColor(resized, cv.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    binary2 = np.zeros_like(v_channel)
    binary2[(v_channel > 150)] = 1

    allBinary = np.zeros_like(binary)
    allBinary[(binary == 1)|(binary2 == 1)] = 255

    # Select the area under the wheels

    src = np.float32([[0, 180],
                      [280, 180],
                      [175, 120],
                      [85, 120]])

    src_draw = np.array(src, dtype=np.int32)

    allBinary_visual = allBinary.copy()
    cv.polylines(allBinary_visual, [src_draw], True, 255)
    cv.imshow('under wheels', allBinary_visual)

    # convert trapezoid to rectangle
    dst = np.float32([[0, img_size[0]],
                      [img_size[1], img_size[0]],
                      [img_size[1], 0],
                      [0, 0]])

    # transform matrix
    M = cv.getPerspectiveTransform(src, dst)
    warped = cv.warpPerspective(allBinary, M, (img_size[1], img_size[0]), flags=cv.INTER_LINEAR)
    cv.imshow('the area', warped)


    # find vertical lines
    histogram = np.sum(warped[warped.shape[0]//2:, :], axis=0)


    midpoint = histogram.shape[0]//2
    indL = np.argmax(histogram[:midpoint])
    indR = np.argmax(histogram[midpoint:]) + midpoint
    warped_visual = warped.copy()
    cv.line(warped_visual, (indL, 0), (indL, warped_visual.shape[0]), 110, 2)
    cv.line(warped_visual, (indR, 0), (indR, warped_visual.shape[0]), 110, 2)


    nwindows = 9
    window_height = np.int(warped.shape[0] / nwindows)
    window_width = 25

    XCenterL = indL
    XCenterR = indR

    left_lane_ind = np.array([], dtype=np.int16)
    right_lane_ind = np.array([], dtype=np.int16)

    out_img = np.dstack((warped, warped, warped))

    nonzero = warped.nonzero()
    WhitePixelIndY = np.array(nonzero[0])
    WhitePixelIndX = np.array(nonzero[1])

    for window in range(nwindows):
        win_y1 = warped.shape[0] - (window + 1) * window_height
        win_y2 = warped.shape[0] - (window) * window_height

        left_win_x1 = XCenterL - window_width
        left_win_x2 = XCenterL + window_width
        right_win_x1 = XCenterR - window_width
        right_win_x2 = XCenterR + window_width

        cv.rectangle(out_img, (left_win_x1, win_y1), (left_win_x2, win_y2), (50 + window * 21, 0, 0), 2)
        cv.rectangle(out_img, (right_win_x1, win_y1), (right_win_x2, win_y2), (0, 0, 50 + window * 21), 2)

        good_left_inds = ((WhitePixelIndY>win_y1) & (WhitePixelIndY<=win_y2) & (WhitePixelIndX>=left_win_x1) & (WhitePixelIndX<=left_win_x2)).nonzero()[0]

        good_right_inds = ((WhitePixelIndY>win_y1) & (WhitePixelIndY<=win_y2) & (WhitePixelIndX>right_win_x1) & (WhitePixelIndX<=right_win_x2)).nonzero()[0]

        left_lane_ind = np.concatenate((left_lane_ind, good_left_inds))
        right_lane_ind = np.concatenate((right_lane_ind, good_right_inds))

        if len(good_left_inds) > 50:
            XCenterL = np.int(np.mean(WhitePixelIndX[good_left_inds]))
            XCenterR = np.int(np.mean(WhitePixelIndX[good_right_inds]))



    out_img[WhitePixelIndY[left_lane_ind], WhitePixelIndX[left_lane_ind]] = [255, 0, 0]
    out_img[WhitePixelIndY[right_lane_ind], WhitePixelIndX[right_lane_ind]] = [0, 0, 255]

    leftx = WhitePixelIndX[left_lane_ind]
    lefty = WhitePixelIndY[left_lane_ind]
    rightx = WhitePixelIndX[right_lane_ind]
    righty = WhitePixelIndY[right_lane_ind]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit =  np.polyfit(righty, rightx, 2)

    center_fit = ((left_fit+right_fit)/2)

    for ver_ind in range(out_img.shape[0]):
        gor_ind = (center_fit[0]) * (ver_ind ** 2) + center_fit[1] * ver_ind + center_fit[2]

        cv.circle(out_img, (int(gor_ind), int(ver_ind)), 2, (0, 150, 0), 1)
    cv.imshow('road_line', out_img)



