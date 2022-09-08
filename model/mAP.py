import numpy as np
from typing import List, Dict, Tuple
from dataset import data_set_builder

dataset = data_set_builder()
#print(dataset[0]['img_file_name'])
#print(dataset[0]['img'])
print(dataset[0]['boxes_labelled']['labels'])
#print(dataset[0]['target_file_name'])



