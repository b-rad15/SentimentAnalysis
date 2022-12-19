import re

import predict_all_urls
from ModelProcessing import ModelProcessing
import helper
import numpy as np
from matplotlib import pyplot as plt
import subprocess
import sys, os

model_object = ModelProcessing(yaml_file="model-cuda.yaml")
ign_regex = re.compile(r"^\s*(?P<protocol>https?://)?(?P<url>(?P<subdomain>www\.)?ign\.com/articles/(?P<ymd>(?P<year>[0-9]{4})/(?P<month>[0-9]{2})/(?P<day>[0-9]{2})/)?(?P<article_name>[^/]+))/?\s*$")


while (url := input("Enter an IGN Review URL (enter \"quit\" to exit): ")) != "quit":
    match = ign_regex.match(url)
    if not match:
        print(f"Article {url} does not match the format ign.com/articles/article-name ({ign_regex.pattern})")
        continue
    url = match["url"]
    sentences = predict_all_urls.get_text_from_ign_url(url)
    predictions = model_object.predict_messages(sentences)
    emotion_vector = np.sum(predictions, axis=0)
    emotion_vector = emotion_vector / np.linalg.norm(emotion_vector)
    helper.print_confidences(emotion_vector, model_object.output_dict)
    fig: plt.Figure = predict_all_urls.make_bar_chart(emotion_vector, model_object.output_dict)
    chart_file = match["article_name"] + ".svg"
    plt.savefig(chart_file)
    print(f"Chart saved to {os.getcwd()}/{chart_file}")
    plt.show()
    plt.close(fig)
    try:
        # Try to open chart with system specific open command
        if sys.platform.startswith('win32') or sys.platform.startswith('cygwin'):
            subprocess.run(["explorer.exe", chart_file])
        elif sys.platform.startswith('darwin'):
            subprocess.run(["open", chart_file])
        else:
            subprocess.run(["xdg-open", chart_file])
    except:
        # failing is unimportant as long as it doesn't crash the program
        pass
