import os
import importlib.resources

os.environ["NLTK_DATA"] = str(importlib.resources.files("SongBloom.g2p.cn_zh_g2p") / "nltk_data")
