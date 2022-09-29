import PySimpleGUI as sg
import tensorflow as tf
import numpy as np
import os
import shutil
import cv2
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



#############################################################################################
# Name          : Plot_PredVal
# Description   : This API plots the predicted values by the Keras APIs
# Input         : Predicted values, expected output
# Output        : Figure
#############################################################################################

def Plot_PredVal(predictions_array, expected_label = -1):
  plt.figure(figsize=(5, 5))
  plt.grid(False)
  plt.xticks(range(num_classes),ds_class_names)
  thisplot = plt.bar(range(num_classes), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
   
  plt.ylabel('Probability')
  plt.xlabel('Kitchenware Class')
  thisplot[predicted_label].set_color('red')
  if(expected_label != -1): 
    thisplot[expected_label].set_color('green')
      
  return plt.gcf()
  

#############################################################################################
# Name          : Draw_figCanvas
# Description   : This API displays the plot on GUI
# Input         : Canvas, Matplotlib figure
# Output        : Figure
#############################################################################################

def Draw_figCanvas(canvas, figure):

    ## To remove the previous plot
    if(fig_agg != None):
        fig_agg.get_tk_widget().forget()
        try:
            draw_figure.canvas_packed.pop(fig_agg.get_tk_widget())
        except Exception as e:
            print(f'Error removing {fig_agg} from canvas', e)
        plt.close('all')
    
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)    
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg
    
    
#############################    M A I N    S E C T I O N   ######################################

####
# Global values
####

## Desired image resolution
img_height = 256
img_width = 256
batch_size = 32
epochs_train = 5
num_classes = 3
dataset_foldername = 'data'
fig_agg = None
cd_path = os.path.dirname(os.path.abspath(__file__)) #Current path


####
# GUI
####
column_1 = [
    [
        sg.Text("Select the image"),
        sg.In(size=(25, 1), enable_events=True, key="IMAGE_PRED"),
        sg.FileBrowse(),
    ],
    [
        sg.Text("", size=(0, 1), key='PREDICTION_RES')
    ],
    [
        sg.OptionMenu(  values=['No feedback','Prediction is correct','Prediction should have been "cups"','Prediction should have been "dishes"','Prediction should have been "plates"',],
                        size=(30, 1), 
                        visible=False,
                        default_value='No feedback',key='FEEDBACK_SEL'),
        sg.Checkbox('Update Model', default=False,visible=False, key="FEEDBACK_MODEL"),
        sg.Button('Submit Feedback', enable_events=True, visible=False, key='FEEDBACK_TRIGGER', size=(0, 1)),
    ],
    [
        sg.Text("Ready!", size=(0, 1), key='MODEL_TRAIN')
    ],
]

column_2 = [
    [sg.Canvas(size=(500, 500), key='PLOT_CANVAS')]
]

## Full Layout
layout = [
    [
        sg.Column(column_1),
        sg.VSeperator(),
        sg.Column(column_2),
    ]
]

window = sg.Window("Kitchenware Classifier", layout)


###
# Prepare the model
###
data_path_train = cd_path + '/' + dataset_foldername + '/'  + 'train'
data_path_test = cd_path + '/' +  dataset_foldername + '/'  + 'test'

ds_train = tf.keras.utils.image_dataset_from_directory(
    data_path_train,
    color_mode = 'rgb',
    image_size=(img_height,img_width), # reshape
    shuffle = True,
    batch_size = batch_size,
    validation_split=0.2,
    subset="training",
    seed=100
)

ds_val = tf.keras.utils.image_dataset_from_directory(
    data_path_train,
    color_mode = 'rgb',
    image_size=(img_height,img_width), # reshape
    shuffle = True,
    batch_size = batch_size,
    validation_split=0.2,
    subset="validation",
    seed=100
)
  
ds_test = tf.keras.preprocessing.image_dataset_from_directory(
    data_path_test,
    color_mode = 'rgb',
    image_size=(img_height,img_width), # reshape
    shuffle = True,
    batch_size = batch_size,
    seed=100
)

# Read the class list from train or test set
ds_class_names = ds_test.class_names
print(ds_class_names)

## Load the weights
model = tf.keras.models.load_model(cd_path + '/TrainedModel/kitchenware_trainedmodel.h5')

###
# Run the Loop. It checks for the events in the GUI
###
while True:
    ## Read events and values from GUI
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
        
    
    ## Predict the dishware image given by user
    if event == "IMAGE_PRED":
        image_path = values["IMAGE_PRED"]
        
        window['MODEL_TRAIN'].update(value='')
        window['PREDICTION_RES'].update(value='Prediction : Wait')
        window.refresh()
        
        ## Prepare the image
        img_ToPred = cv2.imread(image_path)
        img_ToPred = cv2.resize(img_ToPred, (img_width,img_height))
        img_ToPred = np.expand_dims(img_ToPred, axis=0)

        ## Predict the image using model
        predictions = model.predict(img_ToPred)
        print(predictions)
        predicted_text = 'Prediction : The image is of category "' + ds_class_names[np.argmax(predictions)] +'"'
        
        window['PREDICTION_RES'].update(value=predicted_text)
        window['FEEDBACK_SEL'].update(visible=True)
        window['FEEDBACK_TRIGGER'].update(visible=True)
        window['FEEDBACK_MODEL'].update(visible=True)
         
        predictions = np.squeeze(predictions)
                
        fig_agg = Draw_figCanvas(window['PLOT_CANVAS'].TKCanvas, Plot_PredVal(predictions))       

    ## Check for feedback
    elif event == "FEEDBACK_TRIGGER":
        
        user_feedback = values["FEEDBACK_SEL"]
        user_fbModelUpdate = values["FEEDBACK_MODEL"]
        
        expected_prediction = 999
        if(user_feedback == 'Prediction is correct'):
            expected_prediction = np.argmax(predictions)
        elif(user_feedback == 'Prediction should have been "cups"'):
            expected_prediction = 0
        elif(user_feedback == 'Prediction should have been "dishes"'):
            expected_prediction = 1
        elif(user_feedback == 'Prediction should have been "plates"'):
            expected_prediction = 2

        ## Incase of wrong prediction and user wants to update model
        if( expected_prediction >= 0 and  expected_prediction <= 2):

            fig_agg = Draw_figCanvas(window['PLOT_CANVAS'].TKCanvas, Plot_PredVal(predictions,expected_prediction))
                      
            window['FEEDBACK_SEL'].update(visible=False)
            window['FEEDBACK_TRIGGER'].update(visible=False)
            window['FEEDBACK_MODEL'].update(visible=False)
            window['MODEL_TRAIN'].update(value='Please wait...')  
            window.refresh()
            
            ## User wants to update the model 
            if(user_fbModelUpdate == True):
            
                temp_UserPic_path = cd_path + '/userInput'   
                # Empty the folder and create the folder structure
                if(os.path.exists(cd_path + '/userInput') == True):
                    shutil.rmtree(cd_path + '/userInput')
                    
                if(os.path.exists(temp_UserPic_path) == False):
                    os.makedirs(temp_UserPic_path)
                    os.makedirs(temp_UserPic_path + '/cups')
                    os.makedirs(temp_UserPic_path + '/dishes')
                    os.makedirs(temp_UserPic_path + '/plates')
                
                shutil.copy(image_path, temp_UserPic_path + '/' + ds_class_names[expected_prediction] )
                            
                ds_UserData = tf.keras.preprocessing.image_dataset_from_directory(
                    temp_UserPic_path,
                    color_mode = 'rgb',
                    image_size=(img_height,img_width), # reshape
                    shuffle = False,
                    batch_size = 1,
                    seed=100
                )
            
                window['MODEL_TRAIN'].update(value='Please wait till the model is trained...')  
                window.refresh()
                
                history = model.fit(ds_UserData, validation_data=ds_val, epochs=epochs_train)
                predictions = model.predict(img_ToPred)
                print(predictions)
                
        window['MODEL_TRAIN'].update(value='Ready!')            
        window['PREDICTION_RES'].update(value='')
        window['FEEDBACK_SEL'].update(visible=False)
        window['FEEDBACK_TRIGGER'].update(visible=False)
        window['FEEDBACK_MODEL'].update(visible=False)
        # window.refresh()
            
            
            

## Rewrite the model so that next time it gets preloaded
model.save(cd_path + '/TrainedModel/kitchenware_trainedmodel.h5')

window.close()