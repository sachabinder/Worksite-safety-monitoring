# Worksite safety monitoring solution
To clone the repository : `git clone https://github.com/sachabinder/Worksite-safety-monitoring.git`

## Execute the learning script

Execute learning script require a lot of computation power. That is why, we suggest to run it on [Google collab](colab.research.google.com).
* On your own installation (you need to run first the [environement setup](environment_setup.sh)):
 `python -m model.training`
 * On collab, open the file [training script](training_script.ipymb) build for it.
 
  To get the monitorning in real time with [Neptune API](neptune.ai), personalize the `NEPTUNE_API_KEY` and `NEPTUNE_PROJECT_NAME`.