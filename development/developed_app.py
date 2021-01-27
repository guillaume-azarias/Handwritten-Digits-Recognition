import omegaml as om
import json # No need to specify in the setup file (NNtS)

import dash # (NNtS)
import dash_bootstrap_components as dbc  # Specify in the setup file ?
import dash_core_components as dcc  # (NNtS)
import dash_html_components as html  # (NNtS)
import numpy as np  # (NNtS)
import omegaml as om  # (NNtS)
import plotly.express as px  # (NNtS)
import sklearn  # Specify in the setup file ?
from sklearn import svm  # Specify in the setup file ?
import tensorflow as tf  # Specify in the setup file ?
from dash.dependencies import Input, Output  # (NNtS)
from dash_canvas import DashCanvas  # Specify in the setup file
from skimage import io  # Specify in the setup file

from prepare_input import preprocess

# assert sklearn.__version__ == '0.21.3', 'The sklearn version must be 0.21.3'

print('Restarting the dashboard')


# Retrieve the trained models
# assert len(models) > 0, 'You must first generate models !'
# current_model = models[0]
# clf = om.models.get(current_model)
# clf = om.runtime.model(current_model)

# Canvas parameters
input_size = 28
magnification = 8 # Must be an integer
bord_size = 5  # This is where you set the size of the bord
canvas_width = magnification*input_size
predicted = ""

# Initial fig
img = np.full((100, 100), 255, dtype=int)
img[99,99] = 0
fig = px.imshow(img, color_continuous_scale='gray')
fig.update_layout(width=420, height=400, coloraxis_showscale=False,
paper_bgcolor='rgba(0,0,0,0)',
plot_bgcolor='rgba(0,0,0,0)'
                )
fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)


# ------------------------------------------------------------------------------
# App layout

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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
                ),
                dbc.Col(
                    html.A(
                        html.Img(
                            id="dash_logo", src="https://github.com/guillaume-azarias/Models/blob/master/dash-logo-new.jpg?raw=true",
                            style={"float": "right", "height": 80}
                        ), href='https://plotly.com/')
                )
            ],
            justify="between",
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
                        html.P("Identify yourself:"),
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
                        html.P("Select your own model:"),
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
                    ], md = 4
                ),
                dbc.Col(
                    [
                        html.P("Input pre-processing:"),
                        dcc.Checklist(
                            id='scale',
                            options=
                                [{'label': ' Scale and center the input', 'value': 'scaled'}],
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
                                    hide_buttons=['Pencil tool', 'line', 'zoom', 'rectangle', 'select', 'pan'],
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
                            marks={i: "{}".format(i) for i in range(10, 28, 4)}
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
                            style={'color': 'black','fontSize': 200,
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
                            "Note that SVM classifiers work better when the input is centered as mentioned by ",
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
        print('not all')
        raise dash.exceptions.PreventUpdate

    try:
        print('userid ' + userid)
        print('apikey ' + apikey)
        import omegaml as om
        # om = om.client.cloud.setup(userid=userid, apikey=apikey)

        models = om.models.list()
        return [{"label": model, "value": model} for model in models]
    except Exception as e:
        print('model import error: %s' % e)

@app.callback(
    Output(component_id='selected_model', component_property='children'),
    Input(component_id='model_from_menu', component_property='value')
)
def select_model(value):
    model_name = value
    return model_name


@app.callback(
    Output(component_id='canvas_image', component_property='lineWidth'),
    Input(component_id='linewidth-slider', component_property='value')
    )
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
    username (string): credential information
    apikey (string): credential information
    selected_model (string): name of the selected model
    value (string): option to scale or not the input
    string (string). drawing captured from the canvas

    output:
    fig: figure, image of the preprocessed canvas input
    predicted: integer, predicted value

    Modified from https://dash.plotly.com/canvas
    '''
    global canvas_width, fig, predicted, clf


    if not selected_model:
        print('model not selected')
        raise dash.exceptions.PreventUpdate
    
    if string:
        try:
            # import omegaml as om
            import omegaml as om
            from omegaml.client.auth import OmegaRestApiAuth
            om = om.client.cloud.setup(
                userid=userid, apikey=apikey)
        except Exception as e:
            print('error: %s' % e)

        # if selected_model != current_model:
        #     print('Loading new model')
        #     current_model=selected_model
        #     # clf = om.models.get(current_model)
        #     clf = om.runtime.model(current_model)
        print('selected model: ' + selected_model)

        clf = om.runtime.model(selected_model)

        # Note that the output of the canvas is already normalized
        processed_input = preprocess(
            string, canvas_width, value, input_size, bord_size)

        # Show the processed input
        fig = px.imshow(processed_input*(-255), color_continuous_scale='gray')
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)

        # Flatten the image for ML prediction
        processed_input = processed_input.reshape(1, -1)

        # cnn acccepts the NOT normalized input
        clf_type = om.models.get(selected_model)
        if isinstance(clf_type, tf.keras.models.Sequential):
            processed_input = processed_input*255

        # Predict
        predicted = clf.predict(processed_input).get()
        print('Model used = ' + str(selected_model))

        # Output for the cnn
        if isinstance(clf_type, tf.keras.models.Sequential):
            result = np.where(predicted == np.amax(predicted))  # for cnn only
            predicted = result[1][0]
            
        print('Predicted: %i' % predicted)

    return fig, predicted


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
