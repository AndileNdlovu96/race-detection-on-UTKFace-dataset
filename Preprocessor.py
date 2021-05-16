import os
import cv2
import  numpy as np
import time
from sklearn.externals import joblib
#from dlib import rectangle
#import dlib
import FeatureExtraction as ft
from FeatureExtraction import LBP

faceEthnicity = ['White', 'Black', 'Asian', 'Indian', 'Other']
input_dir = 'data/UTKFace/'
output_dir = 'data/output/'
preprocess_dir = 'data/preprocess/'


def imageColour(colour, faceImg):
    if colour == 'gray':
        return cv2.cvtColor(faceImg, cv2.COLOR_BGR2GRAY)
    elif colour == 'hsv':
        return cv2.cvtColor(faceImg, cv2.COLOR_BGR2HSV)
    elif colour == 'ycrbr':
        return cv2.cvtColor(faceImg, cv2.COLOR_BGR2YCrCb)
    else:
        return faceImg


predictorPath = "faceModels/shape_predictor_68_face_landmarks.dat"
#detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor(predictorPath)


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
    return ((np.max(feature[:, 0]) + np.min(feature[:, 0]))//2, (np.max(feature[:, 1]) + np.min(feature[:, 1]))//2)

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


def preprocess():
    #'''
    for x in range(len(faceEthnicity)):
        #print('x',faceEthnicity[x])
        os.makedirs(preprocess_dir+faceEthnicity[x]+'/', exist_ok=True)
    faces = os.listdir(input_dir)

    #White, Black, Asian, Indian, and Others
    #faceEthnicity = {0:'White', 1:'Black', 2:'Asian', 3:'Indian', 4:'Other'}
    #print(len(faces))
    faultyPreprosessing = 1
    for i, face in enumerate(faces):
        if face[-3:].lower() != 'jpg':
            if face[-3:].lower() != 'png':
                continue
        facePath = input_dir+face
        faceImg = cv2.imread(facePath)#.astype('float32')
        #faceImg = cv2.resize(faceImg, dsize=(100, 100))
        faceImg = imageColour('rgb', faceImg)

        info = face.split('_')

        if len(info)!= 4:
            print(face)
            print('not enough info')
            print('------', i, '-------')
        else:
            print(face)
            ethnicity = int(face.split('_')[2])
            print(ethnicity)
            e = faceEthnicity[ethnicity]
            print(e)
            print('------', i,'-------')

            faces = getFacialLandmarks(faceImg, detector, predictor)

            if faces is not None:
                #faceImg, landmark_image, masked_face, masked_feature_highlights, roiMasks, regions_of_interest
                faceImg = get_preprocesed_face_image(faceImg, faces)[0]
                landmark_image = get_preprocesed_face_image(faceImg, faces)[1]
                masked_face = get_preprocesed_face_image(faceImg, faces)[2]
                masked_feature_highlights = get_preprocesed_face_image(faceImg, faces)[3]
                roiMasks = get_preprocesed_face_image(faceImg, faces)[4]
                regions_of_interest = get_preprocesed_face_image(faceImg, faces)[5]

                print(preprocess_dir + e + '/' + face[:-13] + '/')
                x = preprocess_dir + e + '/' + face[:-13] + '/'
                os.makedirs(x, exist_ok=True)
                faceImg = cv2.resize(faceImg, dsize=(100, 100))
                cv2.imwrite(x + face[:-13] + '__' + "faceImg" + ".jpg", faceImg)
                landmark_image = cv2.resize(landmark_image, dsize=(100, 100))
                cv2.imwrite(x + face[:-13] + '__' + "landmark_image" + ".jpg", landmark_image)
                masked_face = cv2.resize(masked_face, dsize=(100, 100))
                cv2.imwrite(x + face[:-13] + '__' + "masked_face" + ".jpg", masked_face)

                masked_face_feature_highlight = cv2.resize(masked_feature_highlights['Face'], dsize=(100, 100))
                cv2.imwrite(x + face[:-13] + '__' + "masked_face_feature_highlight" + ".jpg",
                            masked_face_feature_highlight)
                masked_mouth_feature_highlight = cv2.resize(masked_feature_highlights['Mouth'], dsize=(100, 100))
                cv2.imwrite(x + face[:-13] + '__' + "masked_mouth_feature_highlight" + ".jpg",
                            masked_feature_highlights['Mouth'])
                masked_nose_feature_highlight = cv2.resize(masked_feature_highlights['Nose'], dsize=(100, 100))
                cv2.imwrite(x + face[:-13] + '__' + "masked_nose_feature_highlight" + ".jpg",
                            masked_feature_highlights['Nose'])
                masked_left_eye_feature_highlight = cv2.resize(masked_feature_highlights['Left Eye'], dsize=(100, 100))
                cv2.imwrite(x + face[:-13] + '__' + "masked_left_eye_feature_highlight" + ".jpg",
                            masked_feature_highlights['Left Eye'])
                masked_right_eye_feature_highlight = cv2.resize(masked_feature_highlights['Right Eye'], dsize=(100, 100))
                cv2.imwrite(x + face[:-13] + '__' + "masked_right_eye_feature_highlight" + ".jpg",
                            masked_feature_highlights['Right Eye'])

                roiFaceMask = cv2.resize(roiMasks['Face'], dsize=(100, 100))
                cv2.imwrite(x + face[:-13] + '__' + "roiFaceMask" + ".jpg", roiFaceMask)
                roiMouthMask = cv2.resize(roiMasks['Mouth'], dsize=(100, 100))
                cv2.imwrite(x + face[:-13] + '__' + "roiMouthMask" + ".jpg", roiMasks['Mouth'])
                roiNoseMask = cv2.resize(roiMasks['Nose'], dsize=(100, 100))
                cv2.imwrite(x + face[:-13] + '__' + "roiNoseMask" + ".jpg", roiNoseMask)
                roiLeftEyeMask = cv2.resize(roiMasks['Left Eye'], dsize=(100, 100))
                cv2.imwrite(x + face[:-13] + '__' + "roiLeftEyeMask" + ".jpg", roiLeftEyeMask)
                roiRightEyeMask = cv2.resize(roiMasks['Right Eye'], dsize=(100, 100))
                cv2.imwrite(x + face[:-13] + '__' + "roiRightEyeMask" + ".jpg", roiRightEyeMask)

                face_region_of_interest = cv2.resize(regions_of_interest['Face'], dsize=(100, 100))
                cv2.imwrite(x + face[:-13] + '__' + "face_region_of_interest" + ".jpg", face_region_of_interest)
                mouth_region_of_interest = cv2.resize(regions_of_interest['Mouth'], dsize=(100, 100))
                cv2.imwrite(x + face[:-13] + '__' + "mouth_region_of_interest" + ".jpg", mouth_region_of_interest)
                nose_region_of_interest = cv2.resize(regions_of_interest['Nose'], dsize=(100, 100))
                cv2.imwrite(x + face[:-13] + '__' + "nose_region_of_interest" + ".jpg", nose_region_of_interest)
                left_eye_region_of_interest = cv2.resize(regions_of_interest['Left Eye'], dsize=(100, 100))
                cv2.imwrite(x + face[:-13] + '__' + "left_eye_region_of_interest" + ".jpg",
                            left_eye_region_of_interest)
                right_eye_region_of_interest = cv2.resize(regions_of_interest['Right Eye'], dsize=(100, 100))
                cv2.imwrite(x + face[:-13] + '__' + "right_eye_region_of_interest" + ".jpg",
                            right_eye_region_of_interest)
            else:
                faultyPreprosessing+=1
                print(i, face)

    print(faultyPreprosessing, 'images could not be preprocessed and therefore do not belong to the dataset to be used')
'''
def makeIntoVector(path, feature):
    feature_path = path+'/'+feature
    #print('[INFO] image path:   ', feature_path)
    x = cv2.imread(feature_path)
    x = np.array(x)
    #print('[INFO] image size:  ',x.shape)
    x = x.flatten()
    #print('[INFO] vector size:  ', x.shape)
    return x,  x.shape

def getFeaturesAndLabels():
    facial_feature = ['Face', 'Mouth', 'Nose', 'Left Eye', 'Right Eye']
    labels = os.listdir(output_dir)
    y = []
    for label in labels:
        y.append(faceEthnicity.index(label))
    #y = y[faceEthnicity.index(i) for i in range(len(eths))]
    y = np.array(y)
    print('shape of y:', y.shape)
    X = [[]for i in y]
    for i, label in enumerate(labels):
        path = output_dir+ label
        #print(path)
        features = os.listdir(path)
        #print(features)
        for feature in features:
            x, shapeOf_x = makeIntoVector(path, feature)
            X[i].append(x)
        print('[INFO] length of features of label', label, ':   ',len(X[i]))
    print('_______________________________')
    print('[INFO] length of X:', len(X))
    print('[INFO] length of y:', len(labels))

    X = np.array(X)
    print('shape of X:',X.shape)
    return X, y

def getXy():
    facial_features = ['Face', 'Mouth', 'Nose', 'Left Eye', 'Right Eye']
    labels = os.listdir(preprocess_dir)
    ys=[faceEthnicity.index(label)for label in labels]
    #y=np.array(y)
    print(ys)
    X = []
    y = []
    for i, label in enumerate(labels):
        path = preprocess_dir + label
        #print(path)
        features = os.listdir(path)
        #print(features)
        fs = []
        shapeOf_x = 0
        for feature in features:
            x, shapeOf_x = makeIntoVector(path, feature)
            fs.append(x)
            y.append(ys[i])
            print(ys[i])
        print(len(fs))
        print(shapeOf_x)
        fs = tuple(fs)
        fs = np.vstack(fs)#[:,:, np.newaxis]
        print('FS now', fs.shape)
        X.append(fs)

        print(len(X))#=5
    X = tuple(X)
    X = np.vstack(X)
    print('X shape now',X.shape)

    y = np.array(y)
    print('y shape', y.shape)
    #X = np.matrix(X)
    #print('X shape now',X.shape)
    return X,y


def loadDataSet():
    lbp = LBP(8, 1)
    faceEthnicity = ['White', 'Black', 'Asian', 'Indian', 'Other']
    input_dir = 'data/UTKFace.tar/UTKFace/'
    # output_dir = 'data/output/'
    preprocess_dir = 'data/preprocess/'
    facial_features2 = ['face_region_of_interest', 'mouth_region_of_interest', 'nose_region_of_interest',
                       'left_eye_region_of_interest', 'right_eye_region_of_interest']

    labels = os.listdir(preprocess_dir)
    #print(labels)
    ys = [faceEthnicity.index(label) for label in labels]
    # y=np.array(y)
    #print(ys)
    X = {'face': None, 'mouth': None, 'nose': None, 'left eye': None, 'right eye': None}
    X_face = []
    X_nose = []
    X_mouth = []
    X_left_eye = []
    X_right_eye = []
    y = {'face': None, 'mouth': None, 'nose': None, 'left eye': None, 'right eye': None}
    y_face = []
    y_nose = []
    y_mouth = []
    y_left_eye = []
    y_right_eye = []
    for i, label in enumerate(labels):
        path_to_label = preprocess_dir + label
        #print(path_to_label)
        individuals = os.listdir(path_to_label)
        #print(individuals)
        for individual in individuals:
            path_to_individual = path_to_label+'/'+individual+'/'
            #print(path_to_individual)
            #print('-----------')
            facial_features = os.listdir(path_to_individual)

            for f in facial_features:
                if f.split('__')[1][:-4] == 'face_region_of_interest':
                    path_to_facial_feature = path_to_individual + f
                    face_lbph = ft.getFeatureVector(path_to_facial_feature, lbp)
                    X_face.append(face_lbph)
                    y_face.append(ys[i])
                elif f.split('__')[1][:-4] == 'mouth_region_of_interest':
                    path_to_facial_feature = path_to_individual + f
                    mouth_lbph = ft.getFeatureVector(path_to_facial_feature, lbp)
                    X_mouth.append(mouth_lbph)
                    y_mouth.append(ys[i])
                elif f.split('__')[1][:-4] == 'nose_region_of_interest':
                    path_to_facial_feature = path_to_individual + f
                    nose_lbph = ft.getFeatureVector(path_to_facial_feature, lbp)
                    X_nose.append(nose_lbph)
                    y_nose.append(ys[i])
                elif f.split('__')[1][:-4] == 'left_eye_region_of_interest':
                    path_to_facial_feature = path_to_individual + f
                    left_eye_lbph = ft.getFeatureVector(path_to_facial_feature, lbp)
                    X_left_eye.append(left_eye_lbph)
                    y_left_eye.append(ys[i])
                elif f.split('__')[1][:-4] == 'right_eye_region_of_interest':
                    path_to_facial_feature = path_to_individual + f
                    right_eye_lbph = ft.getFeatureVector(path_to_facial_feature, lbp)
                    X_right_eye.append(right_eye_lbph)
                    y_right_eye.append(ys[i])

    X_face = np.vstack(tuple(X_face))
    #print('X_face shape now', X_face.shape)
    y_face = np.array(y_face)
    #print('y_face shape', y_face.shape)
    X['face'] = X_face
    y['face'] = y_face

    X_mouth = np.vstack(tuple(X_mouth))
    #print('X_mouth shape now', X_mouth.shape)
    y_mouth = np.array(y_mouth)
    #print('y_mouth shape', y_mouth.shape)
    X['mouth'] = X_mouth
    y['mouth'] = y_mouth

    X_nose = np.vstack(tuple(X_nose))
    #print('X_nose shape now', X_nose.shape)
    y_nose = np.array(y_nose)
    #print('y_nose shape', y_nose.shape)
    X['nose'] = X_nose
    y['nose'] = y_nose

    X_left_eye = np.vstack(tuple(X_left_eye))
    #print('X_left_eye shape now', X_left_eye.shape)
    y_left_eye = np.array(y_left_eye)
    #print('y_left_eye shape', y_left_eye.shape)
    X['left eye'] = X_left_eye
    y['left eye'] = y_left_eye

    X_right_eye = np.vstack(tuple(X_right_eye))
    #print('X_right_eye shape now', X_right_eye.shape)
    y_right_eye = np.array(y_right_eye)
    #print('y_right_eye shape', y_right_eye.shape)
    X['right eye'] = X_right_eye
    y['right eye'] = y_right_eye

    return X, y

def serializeDataset():
    X, y = loadDataSet()
    #joblib.dump(X, 'X.pkl')
    #joblib.dump(y, 'y.pkl')
    joblib.dump({'X':X, 'y':y}, 'X_And_y.pkl')

def deserializeDataSet():
    #return joblib.load('X.pkl'), joblib.load('y.pkl')
    return joblib.load('X_And_y.pkl')

def save_classifier(clf, filename):
    joblib.dump(clf, filename)

def load_classifier(filename):
    return joblib.load(filename)
'''