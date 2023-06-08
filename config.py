import os
import sys

JSON_AS_ASCII = False
MAX_CONTENT_LENGTH = 5242880

# flask debug mode
DEBUG = False
# port
PORT = 23456
# absolute path
ABS_PATH = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])))
# upload path
UPLOAD_FOLDER = ABS_PATH + "/upload"
# cahce path
CACHE_PATH = ABS_PATH + "/cache"
# zh ja ko en... If it is empty then reads from the cleaner
LANGUAGE_AUTOMATIC_DETECT = []
# set to True to enable API Key authentication
API_KEY_ENABLED = False
# API_KEY is required for authentication
API_KEY = "api-key"
# logging_level:DEBUG/INFO/WARNING/ERROR/CRITICAL
LOGGING_LEVEL = "DEBUG"

'''
For each model, the filling method is as follows 模型列表中每个模型的填写方法如下
example 示例:
MODEL_LIST = [
    #VITS
    [ABS_PATH+"/Model/Nene_Nanami_Rong_Tang/1374_epochs.pth", ABS_PATH+"/Model/Nene_Nanami_Rong_Tang/config.json"],
    [ABS_PATH+"/Model/Zero_no_tsukaima/1158_epochs.pth", ABS_PATH+"/Model/Zero_no_tsukaima/config.json"],
    [ABS_PATH+"/Model/g/G_953000.pth", ABS_PATH+"/Model/g/config.json"],
    #HuBert-VITS
    [ABS_PATH+"/Model/louise/360_epochs.pth", ABS_PATH+"/Model/louise/config.json", ABS_PATH+"/Model/louise/hubert-soft-0d54a1f4.pt"],
    #W2V2-VITS
    [ABS_PATH+"/Model/w2v2-vits/1026_epochs.pth", ABS_PATH+"/Model/w2v2-vits/config.json", ABS_PATH+"/all_emotions.npy"],
]
'''
# load mutiple models
MODEL_LIST = [
    # [ABS_PATH+"/model/paimon6k/paimon6k_390000.pth", ABS_PATH+"/model/paimon6k/paimon6k.json"]
    [ABS_PATH+"/model/qiu/G_latest.pth", ABS_PATH+"/model/qiu/moegoe_config.json"]
    # [ABS_PATH+"/model/nahida/G_420000.pth", ABS_PATH+"/model/nahida/config.json"],
    # [ABS_PATH+"/model/test/vits_bert_model.pth", ABS_PATH+"/model/test/bert_vits.json", ABS_PATH+"/model/test/prosody_model.pt"],
]

"""
default params
以下选项是修改VITS 不指定参数时的默认值
"""

# GET 默认音色id
ID = 0
# GET 默认音频格式 可选wav,ogg,silk,mp3
FORMAT = "wav"
# GET 默认语言
LANG = "AUTO"
# GET 默认语音长度，相当于调节语速，该数值越大语速越慢
LENGTH = 1
# GET 默认噪声
NOISE = 0.33
# GET 默认噪声偏差
NOISEW = 0.4
# 长文本分段阈值，max<=0表示不分段,text will not be divided if max<=0
MAX = 50
