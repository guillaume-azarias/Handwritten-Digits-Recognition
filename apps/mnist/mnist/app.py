# Log setup
# https://docs.google.com/presentation/d/e/2PACX-1vT52GLbHrRQpAkNnx698LaW4-2PV7Og52mQoF1kdwSADU9UqGCdwkmGnCdMH4lE24Wgpm2GYO7HUwD0/pub?start=true&loop=false&delayms=10000&slide=id.g8a6e495c85_0_0
from omegaml.client.auth import OmegaRestApiAuth
import logging
import omegaml as om  # (NNtS)
from omegaml.store.logging import OmegaLoggingHandler
logger = om.logger
OmegaLoggingHandler.setup(reset=True)

# Import libraries
import json  # No need to specify in the setup file (NNtS)
from time import sleep

import dash  # (NNtS)
import dash_bootstrap_components as dbc  # Specify in the setup file ?
import dash_core_components as dcc  # (NNtS)
import dash_html_components as html  # (NNtS)
import numpy as np  # (NNtS)
import plotly.express as px  # (NNtS)
import tensorflow as tf  # Specify in the setup file ?
from dash.dependencies import Input, Output  # (NNtS)
from dash_canvas import DashCanvas  # Specify in the setup file
from flask import Blueprint, Response
from skimage import io  # Specify in the setup file

# Do not forget the dot before the module name
from .prepare_input import preprocess

def create_api_blueprint(uri):
    bp = Blueprint('api', __name__, url_prefix=uri)

    @bp.route('/api/test/streaming')
    def streaming():
        def generate():
            for i in range(100):
                sleep(1)
                yield 'waiting'
            yield 'done'

        return Response(generate(), mimetype='text/csv')

    @bp.route('/api/test/blocking')
    def blocking():
        sleep(100)
        return Response('done', mimetype='text/csv')

    return bp


def create_app(context=None, server=None, uri=None, **kwargs):
    """
    the script API execution entry point
    :return: result
    """

    app = dash.Dash(__name__, server=server, url_base_pathname=uri,
                    external_stylesheets=[dbc.themes.BOOTSTRAP])

    # Canvas parameters
    input_size = 28
    magnification = 8  # Must be an integer
    bord_size = 5  # This is where you set the size of the bord
    canvas_width = magnification*input_size
    predicted = ""

    # Initial fig
    img = np.full((100, 100), 255, dtype=int)
    img[99, 99] = 0
    fig = px.imshow(img, color_continuous_scale='gray')
    fig.update_layout(width=420, height=400, coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)

    # ------------------------------------------------------------------------------
    # Layout documentation https://dash-bootstrap-components.opensource.faculty.ai/docs/components/layout/
    app.layout = dbc.Container(
        [
            # Logos
            dbc.Row(
                [
                    html.Br(),
                    html.Br(),
                    dbc.Col(
                        html.A(
                            html.Img(
                                id="om_logo", src="https://hub.omegaml.io/static/logo.jpg",
                                style={"float": "left", "height": 70}
                            ), href='https://www.omegaml.io/')
                        )
                ]
            ),
            # Title
            dbc.Row(
                dbc.Col(
                    [
                        html.Hr(),
                        html.H1("MNIST Digits Recognition using Machine Learning",
                                style={'text-align': 'center'}),
                        html.Hr(),
                    ]
                )
            ),
            # Introductory text
            dbc.Row(
                dbc.Col(
                    [
                        html.P(
                            [
                                "This is a live implementation of the MNIST digits recognition problem, using a models trained and deployed by ",
                                html.A(
                                    "omega|ml",
                                    href="https://www.omegaml.io"
                                ),
                                ". The classification algorithms using C-Support Vector Classification and Feed Forward Neural Network were created following the  ",
                                html.A(
                                    "scikit learn tutorial",
                                    href="https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html"
                                ),
                                " and ",
                                html.A(
                                    "Keras TensorFlow tutorial",
                                    href="https://www.tensorflow.org/tutorials/quickstart/beginner"
                                ),
                                ", respectively. The models have been trained with and deployed by omega|ml with just one line of code.",
                            ],
                        ),
                    ], md=12
                ),
            ),
            # Parameters
            dbc.Row(
                [
                    dbc.Col(
                    [
                        html.P("1) Identify yourself:"),
                        dcc.Input(id="userid", type="text",
                                  placeholder="userid"),
                        dcc.Input(id="apikey", type="password",
                                  placeholder="apikey"),
                        html.P(
                            [
                                "Go to ",
                                html.A(
                                    "your account",
                                    href="https://hub.omegaml.io/profile/user/"
                                    ),
                            ],
                            ),
                        ], md=4
                ),
                dbc.Col(
                        [
                            html.P("2) Select your own model:"),
                            dcc.Dropdown(
                                id='model_from_menu',
                                # options=[
                                #     {"label": model, "value": model} for model in models
                                # ],
                                # value=models[0],
                                clearable=False
                            ),
                            html.Br(),
                            # Hidden div inside the app that stores the intermediate value
                            html.Div(id='selected_model',
                                     style={'display': 'none'})
                        ], md=4
                    ),
                    dbc.Col(
                        [
                            html.P("3) Choose image pre-processing:"),
                            dcc.Checklist(
                                id='scale',
                                options=[
                                    {'label': ' Scale and center the input', 'value': 'scaled'}],
                                value=['scaled'],
                                style={'marginTop': 20, 'margin-left': '10%'}
                            )
                        ], md=4
                    ),
                ],
            ),
            # Body
            dbc.Row(
                [
                    dbc.Col(
                        [
                            # Canvas
                            html.P("Draw a digit in the blue box, then click predict",
                                   style={'text-align': 'center'}),
                            html.Br(),
                            html.Br(),
                            html.Br(),
                            html.Div(
                                DashCanvas(id='canvas_image',
                                           tool='Pencil tool',
                                           lineColor='black',
                                           #    filename=filename, # Would be an option to show a background
                                           width=canvas_width,
                                           height=canvas_width,
                                           hide_buttons=[
                                               'Pencil tool', 'line', 'zoom', 'rectangle', 'select', 'pan'],
                                           goButtonTitle='Predict',
                                           ),
                                style={"display": "block",
                                       "margin-left": '25%'}
                            ),
                            # Slider
                            html.P("Linewidth"),
                            dcc.Slider(
                                id='linewidth-slider',
                                min=10,
                                max=26,
                                step=4,
                                value=18,
                                marks={i: "{}".format(i)
                                       for i in range(10, 28, 4)}
                            ),
                        ], md=4
                    ),
                    dbc.Col(
                        [
                            # Preprocessed Input
                            html.P("Scaled input used by submitted to the model",
                                   style={'text-align': 'center'}),
                            html.Div(
                                dcc.Graph(
                                    id='my-image',
                                    figure=fig,
                                    config={'displayModeBar': False,
                                            'autosizable': True}),
                            ),
                        ],
                        md=4),
                    dbc.Col(
                        [
                            # Predicted value
                            html.P("Value predicted by the model",
                                   style={'text-align': 'center'}),
                            html.Br(),
                            html.Div(
                                predicted,
                                id='predicted_value',
                                style={'color': 'black', 'fontSize': 200,
                                       'text-align': 'center',
                                       'marginTop': 25}
                            ),
                        ],
                        md=4),  # , style={'marginBottom': 50, 'marginTop': 25}
                ],
            ),
            # Further information
            dbc.Row(
                dbc.Col(
                    [
                        html.Br(),
                        html.P(
                            [
                                "Note that SVM classifiers work better when the input is centered, as mentioned by ",
                                html.A(
                                    "Yann Lecun",
                                    href="http://yann.lecun.com/exdb/mnist/"
                                ),
                                " . Therefore, the bounding boxes are computed to center the images following this ",
                                html.A(
                                    "Scipy lecture note",
                                    href="https://scipy-lectures.org/advanced/image_processing/auto_examples/plot_find_object.html"
                                ),
                                ". May you find any discrepancies between the digits you draw and the predicted digit, see below a few examples of the digits used for training (",
                                html.A(
                                    "Lecun et al, 1998",
                                    href="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf"
                                ),
                                "):"
                            ],
                            style={'font-style': 'italic'}
                        ),
                        html.Img(
                            id="mnist_examples", src="https://github.com/guillaume-azarias/Models/blob/master/mnist_examples.jpg?raw=true",
                            style={
                                "display": "block",
                                "margin-left": "auto",
                                "margin-right": "auto",
                                "height": 300}
                        ),
                    ], md=12
                ),
            ),
        ], fluid=True
    )

    # ------------------------------------------------------------------------------
    # Callbacks
    @app.callback(
        Output(component_id='model_from_menu', component_property='options'),
        [Input(component_id='userid', component_property='value'),
         Input(component_id='apikey', component_property='value'),
         Input(component_id='model_from_menu', component_property='search_value')]
    )
    def load_models(userid, apikey, search_value):
        # Handling missing credential
        if not all([userid, apikey]) or len(apikey) < 20:
            raise dash.exceptions.PreventUpdate

        # Loading user models
        try:
            import omegaml as om
            om = om.client.cloud.setup(userid=userid, apikey=apikey)
            models = om.models.list()
            return [{"label": model, "value": model} for model in models]
        except Exception as e:
            logger.error('model import error: %s' % e)

    @app.callback(
        Output(component_id='canvas_image', component_property='lineWidth'),
        Input(component_id='linewidth-slider', component_property='value'))
    def update_canvas_linewidth(value):
        return value

    @app.callback(
        [Output(component_id='my-image', component_property='figure'),
         Output(component_id='predicted_value', component_property='children')],
        [Input(component_id='userid', component_property='value'),
         Input(component_id='apikey', component_property='value'),
         Input(component_id='model_from_menu', component_property='value'),
         Input(component_id='scale', component_property='value'),
         Input(component_id='canvas_image', component_property='json_data')],
        prevent_initial_update=True
    )
    def predict_digit(userid, apikey, selected_model, value, string):
        '''
        Pre-process the data and generate the prediction
        
        input:
        userid (string): omegaml userid
        apikey (string): omegaml apikey
        selected_model (string): name of the selected model
        value (string): option to scale or not the input
        string (string). drawing captured from the canvas

        output:
        fig: figure, image of the (preprocessed) canvas input
        predicted: integer, predicted value

        Modified from https://dash.plotly.com/canvas
        '''

        if not selected_model:
            raise dash.exceptions.PreventUpdate

        if string:
            # Import the model
            import omegaml as om
            om = om.client.cloud.setup(userid=userid, apikey=apikey)
            try:
                clf = om.runtime.model(selected_model)
                logger.info('selected_model imported')
            except Exception as e:
                logger.error('model import error: %s' % e)
                raise dash.exceptions.PreventUpdate

            # Preprocessing. Note that the output of the canvas is already normalized
            processed_input = preprocess(
                string, canvas_width, value, input_size, bord_size)

            # Show the processed input
            fig = px.imshow(processed_input*(-255),
                            color_continuous_scale='gray')
            fig.update_layout(coloraxis_showscale=False)
            fig.update_xaxes(showticklabels=False).update_yaxes(
                showticklabels=False)

            # cnn acccepts the NOT normalized input
            clf_type = om.models.get(selected_model)
            if isinstance(clf_type, tf.keras.models.Sequential):
                processed_input = processed_input * 255

            # Prediction
            processed_input = processed_input.reshape(1, -1)
            try:
                predicted = clf.predict(processed_input).get()
            except Exception as e:
                logger.error('prediction issue: %s' % e)
                predicted = 'Error'

            # Specific output format for the output for the cnn
            if isinstance(clf_type, tf.keras.models.Sequential):
                result = np.where(predicted == np.amax(predicted))
                predicted = result[1][0]

        return fig, predicted

    # End of the mnist app

    bp = create_api_blueprint(uri)
    app.server.register_blueprint(bp)

    return app


if __name__ == '__main__':
    app = create_app(server=True)
    app.run_server()
