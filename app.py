import os
import sys
from utils.merge import merge_model

text = sys.argv[1]

tts = merge_model([
    ["model/G_latest.pth", "model/moegoe_config.json"]
])

output = tts.vits_infer({'text': text, 'id': 0, 'format': 'wav', 'length': 1.0, 'noise': 0.33, 'noisew': 0.4, 'max': 50, 'lang': 'AUTO', 'speaker_lang': ['zh']})

with open('/Users/xiaozhihua/ai/vits-simple-api/test.wav', 'wb') as f:
    f.write(output.getbuffer())
