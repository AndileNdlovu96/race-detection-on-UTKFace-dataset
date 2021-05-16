import dlib
import cv2
import numpy as np
from dlib import rectangle
import dlib
import FeatureExtraction1 as ft
from sklearn.externals import joblib

predictorPath = "faceModels/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictorPath)

def extractFacialLandmarks(faceROI, img, imgScale, predictor):
    upscaledFaceROI = rectangle(int(faceROI.left() / imgScale), int(faceROI.top() / imgScale),
                                int(faceROI.right() / imgScale), int(faceROI.bottom() / imgScale))


    # predict facial landmark points
    facialLandmarks = predictor(img, upscaledFaceROI)

    # make an array of the landmark points with 68 (x,y) coordinates
    facialLandmarkCoords = np.array([[p.x, p.y] for p in facialLandmarks.parts()])

    # transpose the landmark points so that we deal with a 2xn and not an nx2 model, it makes
    # calculations easier along the way when its a row for x's and a row for y's
    return facialLandmarkCoords.T

def downScaleImg(img, imgScale, maxImgSizeForDetection):
    scaledImg = img
    if max(img.shape) > maxImgSizeForDetection:
        imgScale = maxImgSizeForDetection / float(max(img.shape))
        scaledImg = cv2.resize(img, (int(img.shape[1] * imgScale), int(img.shape[0] * imgScale)))

    return scaledImg, imgScale

def getFacialLandmarks(textureImage, detector, predictor, maxImgSizeForDetection=640):
    imgScale = 1
    downScaledImg, imgScale = downScaleImg(textureImage, imgScale, maxImgSizeForDetection)

    # detect face on smaller image (much faster)
    detectedFacesROI = detector(downScaledImg, 1)

    # return nothing if no faces are found
    if len(detectedFacesROI) == 0:
        return None

    # list of facial landmarks for each face in the mapped image
    facialLandmarksList = []
    for faceROI in detectedFacesROI:
        facialLandmarks = extractFacialLandmarks(faceROI, textureImage, imgScale, predictor)
        facialLandmarksList.append(facialLandmarks)
    # return list of faces
    return facialLandmarksList

def reshape_for_polyline(array):
    # do not know what the outer dimension is, but make it an 1x2 now
    return np.array(array, np.int32).reshape((-1, 1, 2))

def drawImposterLandmarks(frame, landmarks, black_image):
    #black_image = np.zeros(frame.shape, np.uint8)

    landmarks = landmarks.T
    jaw = reshape_for_polyline(landmarks[0:17])
    left_eyebrow = reshape_for_polyline(landmarks[22:27])
    right_eyebrow = reshape_for_polyline(landmarks[17:22])
    nose_bridge = reshape_for_polyline(landmarks[27:31])
    lower_nose = reshape_for_polyline(landmarks[30:35])
    left_eye = reshape_for_polyline(landmarks[42:48])
    right_eye = reshape_for_polyline(landmarks[36:42])
    outer_lip = reshape_for_polyline(landmarks[48:60])
    inner_lip = reshape_for_polyline(landmarks[60:68])

    color = (255, 255, 255)
    thickness = 3

    cv2.polylines(black_image, [jaw], False, color, thickness)
    cv2.polylines(black_image, [left_eyebrow], False, color, thickness)
    cv2.polylines(black_image, [right_eyebrow], False, color, thickness)
    cv2.polylines(black_image, [nose_bridge], False, color, thickness)
    cv2.polylines(black_image, [lower_nose], True, color, thickness)
    cv2.polylines(black_image, [left_eye], True, color, thickness)
    cv2.polylines(black_image, [right_eye], True, color, thickness)
    cv2.polylines(black_image, [outer_lip], True, color, thickness)
    cv2.polylines(black_image, [inner_lip], True, color, thickness)

    return black_image


def getCenter(feature):
    return (np.max(feature[:, 0]) + np.min(feature[:, 0]))//2, (np.max(feature[:, 1]) + np.min(feature[:, 1]))//2


def get_preprocesed_face_image(faceImg, faces):
    blank_image = np.zeros(faceImg.shape, np.uint8)
    for facialLandmarks2D in faces:
        # draw the landmarks of all the faces detected from the source frame
        landmark_image = drawImposterLandmarks(faceImg, facialLandmarks2D, blank_image)

        left_eye = facialLandmarks2D.T[42:48]
        center_of_left_eye = getCenter(left_eye)

        right_eye = facialLandmarks2D.T[36:42]
        center_of_right_eye = getCenter(right_eye)

        outer_lip = facialLandmarks2D.T[48:60]
        center_of_mouth = getCenter(outer_lip)

        nose_ridge = facialLandmarks2D.T[27:31]
        center_of_nose_ridge = getCenter(nose_ridge)
        lower_nose = facialLandmarks2D.T[30:35]
        center_of_lower_nose = getCenter(lower_nose)

        center_of_nose = ((center_of_lower_nose[0] + center_of_nose_ridge[0]) // 2,
                          (center_of_lower_nose[1] + center_of_nose_ridge[1]) // 2)

        face_Outline = cv2.convexHull(facialLandmarks2D.T)
        face_mask = np.zeros(faceImg.shape, np.uint8)
        cv2.fillConvexPoly(face_mask, face_Outline, (255, 255, 255))
        masked_face = cv2.bitwise_and(faceImg, face_mask)
        masked_face_feature_highlight = masked_face.copy()
        masked_nose_feature_highlight = masked_face.copy()
        masked_mouth_feature_highlight = masked_face.copy()
        masked_left_eye_feature_highlight = masked_face.copy()
        masked_right_eye_feature_highlight = masked_face.copy()

        masked_feature_highlights = {'Face': None, 'Nose': None, 'Mouth': None, 'Left Eye': None, 'Right Eye': None}
        # ------------------For Face---------------
        cv2.circle(masked_face_feature_highlight, center_of_left_eye, 30, (0, 0, 255), -1)
        cv2.circle(masked_face_feature_highlight, center_of_right_eye, 30, (0, 0, 255), -1)
        cv2.circle(masked_face_feature_highlight, center_of_mouth, 40, (0, 0, 255), -1)
        cv2.circle(masked_face_feature_highlight, center_of_nose, 35, (0, 0, 255), -1)
        masked_feature_highlights['Face'] = masked_face_feature_highlight

        # -----------------For Nose----------------
        cv2.circle(masked_nose_feature_highlight, center_of_nose, 35, (0, 0, 255), -1)
        masked_feature_highlights['Nose'] = masked_nose_feature_highlight

        # -----------------For Mouth---------------
        cv2.circle(masked_mouth_feature_highlight, center_of_mouth, 40, (0, 0, 255), -1)
        masked_feature_highlights['Mouth'] = masked_mouth_feature_highlight

        # -----------------For Left Eye------------
        cv2.circle(masked_left_eye_feature_highlight, center_of_left_eye, 30, (0, 0, 255), -1)
        masked_feature_highlights['Left Eye'] = masked_left_eye_feature_highlight

        # -----------------For Right Eye------------
        cv2.circle(masked_right_eye_feature_highlight, center_of_right_eye, 30, (0, 0, 255), -1)
        masked_feature_highlights['Right Eye'] = masked_right_eye_feature_highlight

        # find just the red part of the image
        # start by making a mask
        roiMasks = {'Face':None, 'Nose': None, 'Mouth': None, 'Left Eye': None, 'Right Eye': None}
        roiFaceMask = cv2.inRange(masked_face_feature_highlight, (0, 0, 255), (0, 0, 255))
        roiMasks['Face'] = roiFaceMask
        roiMouthMask = cv2.inRange(masked_mouth_feature_highlight, (0, 0, 255), (0, 0, 255))
        roiMasks['Mouth'] = roiMouthMask
        roiNoseMask = cv2.inRange(masked_nose_feature_highlight, (0, 0, 255), (0, 0, 255))
        roiMasks['Nose'] = roiNoseMask
        roiLeftEyeMask = cv2.inRange(masked_left_eye_feature_highlight, (0, 0, 255), (0, 0, 255))
        roiMasks['Left Eye'] = roiLeftEyeMask
        roiRightEyeMask = cv2.inRange(masked_right_eye_feature_highlight, (0, 0, 255), (0, 0, 255))
        roiMasks['Right Eye'] = roiRightEyeMask

        regions_of_interest = {'Face':None, 'Nose': None, 'Mouth': None, 'Left Eye': None, 'Right Eye': None}
        face_region_of_interest = cv2.bitwise_and(masked_face, masked_face, mask=roiFaceMask)
        regions_of_interest['Face'] = face_region_of_interest
        mouth_region_of_interest = cv2.bitwise_and(masked_face, masked_face, mask=roiMouthMask)
        regions_of_interest['Mouth'] = mouth_region_of_interest
        nose_region_of_interest = cv2.bitwise_and(masked_face, masked_face, mask=roiNoseMask)
        regions_of_interest['Nose'] = nose_region_of_interest
        left_region_of_interest = cv2.bitwise_and(masked_face, masked_face, mask=roiLeftEyeMask)
        regions_of_interest['Left Eye'] = left_region_of_interest
        right_region_of_interest = cv2.bitwise_and(masked_face, masked_face, mask=roiRightEyeMask)
        regions_of_interest['Right Eye'] = right_region_of_interest

        return faceImg, landmark_image, masked_face, masked_feature_highlights, roiMasks, regions_of_interest

def rectT0BoundingBox(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h

if __name__ == '__main__':

    cap = cv2.VideoCapture(0)
    lbp = ft.LBP(8, 1)

    rf = joblib.load('rf.pkl')
    faceEthnicity = ['White', 'Black', 'Asian', 'Indian', 'Other']
    while True:
        faceImg = cap.read()[1]
        faceImg = cv2.flip(faceImg, 1)
        faces = getFacialLandmarks(faceImg, detector, predictor, 320)
        if faces is not None:
            # faceImg, landmark_image, masked_face, masked_feature_highlights, roiMasks, regions_of_interest
            faceImg = get_preprocesed_face_image(faceImg, faces)[0]
            landmark_image = get_preprocesed_face_image(faceImg, faces)[1]
            masked_face = get_preprocesed_face_image(faceImg, faces)[2]
            masked_feature_highlights = get_preprocesed_face_image(faceImg, faces)[3]
            roiMasks = get_preprocesed_face_image(faceImg, faces)[4]
            regions_of_interest = get_preprocesed_face_image(faceImg, faces)[5]



            '''
            masked_face_feature_highlight = cv2.resize(masked_feature_highlights['Face'], dsize=(100, 100))
            cv2.imshow("masked_face_feature_highlight", masked_face_feature_highlight)
            masked_mouth_feature_highlight = cv2.resize(masked_feature_highlights['Mouth'], dsize=(100, 100))
            cv2.imshow("masked_mouth_feature_highlight", masked_feature_highlights['Mouth'])
            masked_nose_feature_highlight = cv2.resize(masked_feature_highlights['Nose'], dsize=(100, 100))
            cv2.imshow("masked_nose_feature_highlight", masked_feature_highlights['Nose'])
            masked_left_eye_feature_highlight = cv2.resize(masked_feature_highlights['Left Eye'], dsize=(100, 100))
            cv2.imshow("masked_left_eye_feature_highlight", masked_feature_highlights['Left Eye'])
            masked_right_eye_feature_highlight = cv2.resize(masked_feature_highlights['Right Eye'], dsize=(100, 100))
            cv2.imshow("masked_right_eye_feature_highlight", masked_feature_highlights['Right Eye'])
    
            roiFaceMask = cv2.resize(roiMasks['Face'], dsize=(100, 100))
            cv2.imshow("roiFaceMask", roiFaceMask)
            roiMouthMask = cv2.resize(roiMasks['Mouth'], dsize=(100, 100))
            cv2.imshow("roiMouthMask", roiMasks['Mouth'])
            roiNoseMask = cv2.resize(roiMasks['Nose'], dsize=(100, 100))
            cv2.imshow("roiNoseMask", roiNoseMask)
            roiLeftEyeMask = cv2.resize(roiMasks['Left Eye'], dsize=(100, 100))
            cv2.imshow("roiLeftEyeMask", roiLeftEyeMask)
            roiRightEyeMask = cv2.resize(roiMasks['Right Eye'], dsize=(100, 100))
            cv2.imshow("roiRightEyeMask", roiRightEyeMask)
            '''


            face_region_of_interest = cv2.resize(regions_of_interest['Face'], dsize=(200, 200))
            cv2.imshow("face_region_of_interest", face_region_of_interest)
            mouth_region_of_interest = cv2.resize(regions_of_interest['Mouth'], dsize=(200, 200))
            cv2.imshow("mouth_region_of_interest", mouth_region_of_interest)
            nose_region_of_interest = cv2.resize(regions_of_interest['Nose'], dsize=(200, 200))
            cv2.imshow("nose_region_of_interest", nose_region_of_interest)
            left_eye_region_of_interest = cv2.resize(regions_of_interest['Left Eye'], dsize=(200, 200))
            cv2.imshow("left_eye_region_of_interest", left_eye_region_of_interest)
            right_eye_region_of_interest = cv2.resize(regions_of_interest['Right Eye'], dsize=(200, 200))
            cv2.imshow("right_eye_region_of_interest", right_eye_region_of_interest)

            faces = detector(faceImg)
            if len(faces) > 0:
                for face in faces:
                    (x, y, w, h) = rectT0BoundingBox(face)
                    cv2.rectangle(faceImg, (x, y), (x + w, y + h), (0, 255, 0), 1)

                    faceLbph = ft.getFeatureVector(face_region_of_interest, lbp).reshape(1, -1)
                    pred_val = rf.predict(faceLbph)[0]

                    prediction = faceEthnicity[pred_val]

                    cv2.putText(faceImg, prediction, (x, (y + h) - 10), cv2.FONT_HERSHEY_DUPLEX, .4, (255, 255, 255))

            faceImg = cv2.resize(faceImg, dsize=(600, 600))
            cv2.imshow("faceImg", faceImg)
            landmark_image = cv2.resize(landmark_image, dsize=(300, 300))
            cv2.imshow("landmark_image", landmark_image)
            masked_face = cv2.resize(masked_face, dsize=(300, 300))
            cv2.imshow("masked_face", masked_face)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()