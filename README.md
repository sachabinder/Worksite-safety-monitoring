Hackathon ENPC '024
Students : Charles BACQUAERT, Sacha BINDER, Hector MARTINET, Arnaud VIRELIZIER

# Worksite safety monitoring solution
To clone the repository : `git clone https://github.com/sachabinder/Worksite-safety-monitoring.git`

## Execute the learning script

Execute learning script require a lot of computation power. That is why, we suggest to run it on [Google collab](colab.research.google.com).
* On your own installation (you need to run first the [environement setup](environment_setup.sh)):
 `python -m model.training`
 * On collab, open the file [training script](training_script.ipymb) build for it.
 
  To get the monitoring in real time with [Neptune API](neptune.ai), personalize the `NEPTUNE_API_KEY` and `NEPTUNE_PROJECT_NAME`.

## Execute the heatmap script
The image_differentiation() function takes a path and a threshold
    * path is the file path to the image set (e.g. "Detection_Test_Set/Detection_Test_Set_Img/")
    * threshold is the threshold under which two images will be considered coming from the same point of (values around 50000 are recommended) 
    RETURNS : a list of list of <str> of image name. 

Run the heatmap_demo() function to build heatmaps of the training set
These images will be saved in Detection_Train_Set/heatmaps

Run the differentiation_demo() function to obtain a set of list of image names. These lists are clusters of images taken from the same point of view.

Run the detection_demo() function to see rectangles around people identified



