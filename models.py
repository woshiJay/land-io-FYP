import sys
import os
sys.path.append(os.path.abspath('code_loader'))

from BaseCNN import create_model as create_baseCNN
from DenseNet import create_model as create_DenseNet
from Inception import create_model as create_Inception
from ResNet import create_model as create_ResNet
from VGG16 import create_model as create_VGG
from Xception import create_model as create_Xception
from Hybridv1 import create_model as create_Hybridv1
# Import other model functions similarly

MODEL_FUNCTIONS = {
    "BaseCNN": create_baseCNN,
    "DenseNet": create_DenseNet,
    "Inception": create_Inception,
    "ResNet": create_ResNet,
    "VGG": create_VGG,
    "Xception": create_Xception,
    "Hybridv1": create_Hybridv1,
}