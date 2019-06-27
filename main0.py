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

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css','//netdna.bootstrapcdn.com/font-awesome/4.0.3/css/font-awesome.css']

@server.route('/recon_feed')
def recon_feed():
    return Response(gen_rec(vid),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@server.route('/video_feed')
def video_feed():
    return Response(gen(vid),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

app.layout = html.Div([
    html.Div(children=[
      html.H1(children='Sounce',className = 'logo'),
      html.Ul(children=[
        html.A(children=[
            html.Li("How to use")
        ],href="#howtouse"),
        html.A(children=[
            html.Li("Motivation")
        ],href="#motivation"),
        html.A(children=[
            html.Li("Try it out")
        ],href="#use"),
        html.A(children=[
            html.Li("Contacts")
        ],href="#contacts"),
      ],className="navigation")
    ],className="header"),html.Div(children=[
        html.H2("Speak without bounds"),
        html.Button(children=[html.A("Try it out",href="#use")],className="primary")
    ],className="welcome"),
    html.Span(id='howtouse'),
    html.Div(children=[
        html.H2("How to use"),
        html.Div(children=[
            html.H4('1. Go to the “Use” tab'),
             html.H4('2. Start working with the product by clicking the "Start" button'),
             html.H4('3. You will see a window with the camera open, place your hand in a designated place (square)'),
             html.H4('4. Start showing gestures'),
             html.H4('5. The program will display the translation'),
             html.H4('6. After finishing work, click “Finish”'),
             html.H4('7. After closing the camera window, you can see the translated text.'),
        ],className="steps")
    ],className="howtouse"),
     html.Span(id='motivation'),
    html.Div(children=[
        html.H2("Motivation"),
        html.P(children="Today, 360 million people in the world are deaf or hard of hearing. Our product was created to support moral and ethical relations in society. We want to help people with hearing impairments to better communicate in society."),
    ],className="motivation"),   
    html.Span(id='use'),
     html.Div(children=[
    	html.H2('Try it out'),
	    html.Div(children=[
	        html.Img(src="/video_feed",className="vid")
	    ],className="vidCont"),
	    html.Button("Predict", id="in", className="primary lessshadow"),
	    html.Textarea(children=[], id="out",readonly="readonly"),
    ],className="use"),
    
    html.Span(id='contacts'),
    html.Div(children=[
        html.Div(className="contact",children=[
            html.H2("Contact Us"),
            html.Form(method="post",action="#",id="image-form",children=[
                html.Div(className="three_col", children=[
                    dcc.Input(type="text",name="fname",placeholder="First Name",className="default fname"),
                    dcc.Input(type="text",name="lname",placeholder="Last Name",className="default lname")
                ]),
                html.Div(className="one_col email", children=[
                    dcc.Input(type="text",name="email",placeholder="Email Address",className="default email"),
                ]),
                html.Div(className="one_col", children=[
                    html.Textarea(rows="4",cols="20",placeholder="Message"),
                ]),
                html.Div(className="one_col",children=[
                    html.Button(id="submit",type="submit",value="Submit",children="Submit")
                ]),
            ]),
        ]),
        html.Div(className="contactsRight",children=[      
            html.Div(className="adNumMail",children=[
                html.Div(children = [

                ],className = "number"),
                 html.Div(children = [
                     
                ],className = "adress"),
                html.Div(children = [

                ],className = "mail"),
            ]),      
            html.Div(className="social",children=[
                html.Div(children = [
					 html.Span(className="icon facebook")
                ],className = "facebook"),
                 html.Div(children = [
                        html.Span(className="icon instagram")
                ],className = "instagram"),
                html.Div(children = [
                  html.Span(className="icon twitter")
                ],className = "twitter"),
                html.Div(children = [
                	 html.Span(className="icon github")
                ],className = "github"),
            ])
        ])
    ],className="contacts")
])
@app.callback(
    Output('out', 'children'),
    [Input('in', 'n_clicks')])
def predict_click(n_clicks = 0):
    if n_clicks != None and n_clicks >= 0:
        prediction = get_cur(vid)
        n_clicks+=1
        return prediction
    #return "NO"

if __name__ == '__main__':
    app.run_server(debug=False)