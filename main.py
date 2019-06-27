import dash
import dash_core_components as dcc
import dash_html_components as html
import copy
import numpy as np
from keras.models import load_model
from time import sleep
from flask import Flask, Response
import cv2
from threading import Lock
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css','//netdna.bootstrapcdn.com/font-awesome/4.0.3/css/font-awesome.css']

# parameters
cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8  # start point/total width
threshold = 60  # binary threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0
bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)

def predict_rgb_image(img):
    result = gesture_names[model.predict_classes(img)[0]]
    print(result)
    return (result)


def predict_rgb_image_vgg(image):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model.predict(image)
    print(f'pred_array: {pred_array}')
    result = gesture_names[np.argmax(pred_array)]
    print(f'Result: {result}')
    print(max(pred_array[0]))
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    print(result)
    return result, score

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        return image

def tr1(frame):
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    #frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
        (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)

    ret, jpeg = cv2.imencode('.jpg', frame)

    return jpeg.tobytes()

def tr2(frame):

    img = remove_background(frame)
    img = img[0:int(cap_region_y_end * frame.shape[0]),
        int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
    # cv2.imshow('mask', img)

    # convert the image into binary image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)

    ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ret, jpeg = cv2.imencode('.jpg', thresh)
    return jpeg.tobytes()

def gen(camera):
    while True:
        frame = tr1(camera.get_frame())
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def gen_rec(camera):
    while True:
        frame = tr2(camera.get_frame())
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

def get_cur(camera):
    thresh = camera.get_frame()
    target = np.stack((thresh,) * 3, axis=-1)
    print(target)
    target = cv2.resize(target, (100, 100))
    target = target.reshape(1, 100, 100, 3)
    prediction, score = predict_rgb_image_vgg(target)
    return prediction

server = Flask(__name__)
app = dash.Dash(__name__, server=server)
vid = VideoCamera()

@server.route('/recon_feed')
def recon_feed():
    return Response(gen_rec(vid),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@server.route('/video_feed')
def video_feed():
    return Response(gen(vid),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

app.layout = html.Div([
    html.H2("WHATTTT"),
    html.H2("Hi"),
    html.H1("Webcam Test"),
    html.Div(children=[
        html.Img(src="/video_feed")
    ]),
    html.Div(children=[
        html.Img(src="/recon_feed")
    ]),
    html.Button("Setup background", id="in"),
    html.Div(children=[], id="out")
])
@app.callback(
    Output('out', 'children'),
    [Input('in', 'n_clicks')])
def predict_click(n_clicks = 0):
    if n_clicks != None and n_clicks >= 0:
        prediction = get_cur(vid)
        n_clicks+=1
        return prediction
    return "NO"

if __name__ == '__main__':
    app.run_server(debug=True)