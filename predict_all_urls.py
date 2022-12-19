# %%
import json
import requests
from bs4 import BeautifulSoup
import re
from ModelProcessing import ModelProcessing
from keras_preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt


def queryArticleUrls(total=500):
    baseUrl = "www.ign.com"
    start_index = 0
    count = 30
    total_articles = float("inf")
    allArticleUrls = None
    if True:
        while start_index < total_articles and start_index < total:
            url = "https://mollusk.apis.ign.com/graphql"

            path = "/graphql?operationName=ReviewsContentFeed&variables=%7B%22filter%22%3A%22All%22%2C%22region%22%3A%22us%22%2C%22startIndex%22%3A{}%2C%22count%22%3A{}%2C%22editorsChoice%22%3Afalse%2C%22sortOption%22%3A%22Latest%22%7D&extensions=%7B%22persistedQuery%22%3A%7B%22version%22%3A1%2C%22sha256Hash%22%3A%22054065bd3bff634d485cf214b0a9fa9d492328f280d2268f2da05548cb319ef6%22%7D%7D".format(
                start_index, count)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36',
                "authority": "mollusk.apis.ign.com",
                "method": "GET",
                "path": path,
                "scheme": "https",
                "accept": "*/*",
                "accept-encoding": "gzip, deflate, br",
                "accept-language": "en-US,en;q=0.9,ja;q=0.8",
                "apollographql-client-name": "kraken",
                "apollographql-client-version": "v0.13.31",
                "dnt": "1",
                "origin": "https://www.ign.com",
                "referer": "https://www.ign.com/reviews",
                "sec-ch-ua": "\"Not?A_Brand\";v=\"8\", \"Chromium\";v=\"108\", \"Microsoft Edge\";v=\"108\"",
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": "\"Windows\"",
                "sec-fetch-dest": "empty",
                "sec-fetch-mode": "cors",
                "sec-fetch-site": "same-site"
            }

            response = requests.get(url + path, headers=headers)

            if response.status_code != 200:
                print("Request failed with status code: " + str(response.status_code))
                break

            data = response.json()["data"]
            total_articles = data["reviewContentFeed"]["pagination"]["total"]
            if allArticleUrls is None:
                allArticleUrls = [''] * total_articles
            for item, index in zip(data["reviewContentFeed"]["contentItems"], range(start_index, start_index + count)):
                if item["content"]["url"] is None or len(item["content"]["url"]) == 0:
                    print(item)
                if index >= len(allArticleUrls):
                    allArticleUrls.extend(allArticleUrls)
                allArticleUrls[index] = baseUrl + item["content"]["url"]

            start_index = data["reviewContentFeed"]["pagination"]["endIndex"]
            count = min(total_articles - start_index, count)
            print(f"{start_index} of {total_articles} done {start_index / total_articles:.2%}")
    else:
        # kept in case the above method fails. as it sometimes does
        response = requests.get("https://www.ign.com/reviews", headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'})
        soup = BeautifulSoup(response.content, "html.parser")
        articles = soup.select("a.item-body")
        allArticleUrls = [tag.get("href") for tag in articles]

    with open("articles.txt", 'w') as f:
        for i in range(start_index):
            print(f"https://{allArticleUrls[i]}")
            f.write(allArticleUrls[i] + '\n')


def get_text_from_ign_url(url: str):
    url = "https://" + url
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36',
        "authority": "mollusk.apis.ign.com",
        "method": "GET",
        "scheme": "https",
        "accept": "*/*",
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "en-US,en;q=0.9,ja;q=0.8",
        "apollographql-client-name": "kraken",
        "apollographql-client-version": "v0.13.31",
        "dnt": "1",
        "origin": "https://www.ign.com",
        "referer": "https://www.ign.com/reviews",
        "sec-ch-ua": "\"Not?A_Brand\";v=\"8\", \"Chromium\";v=\"108\", \"Microsoft Edge\";v=\"108\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site"
    }

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, features="html.parser")
    paragraphs = soup.select("p")
    allSentences = []
    for paragraph in paragraphs:
        sentences = re.split(r'\. |\s]', paragraph.getText(separator="\n"))
        allSentences.extend(sentences)
    return allSentences


def make_bar_chart(emotion_vector, output_dict: dict):
    data = {}
    for emotion, index in output_dict.items():
        data[emotion] = emotion_vector[index]
    fig = plt.figure(figsize=(10, 10))
    plt.bar(data.keys(), data.values())
    plt.xlabel("Emotion")
    plt.ylabel("Relative Amount in Article")
    plt.title("Emotions in Article")
    return fig


# %%
if __name__ == "__main__":
    with open("articles.txt", 'r') as f:
        urls = f.readlines()
    url_and_results = dict()
    if(not os.path.exists("articles.txt")):
        queryArticleUrls(500)
    # %%
    model_object = ModelProcessing(yaml_file="model-cuda.yaml")
    for url, index in zip(urls[0:500], range(500)):
        sentences = get_text_from_ign_url(url.strip())
        predictions = model_object.predict_messages(sentences)
        emotion_vector = np.sum(predictions, axis=0)
        emotion_vector = emotion_vector / np.linalg.norm(emotion_vector)
        url_and_results[url] = emotion_vector
        print(f"{index/100:.0%}")
    # %%
    all_emotions = np.array(list(url_and_results.values()))
    highest_emotions = all_emotions.argmax(axis=0)
    most_joyful = urls[highest_emotions[model_object.output_dict["joy"]]]
    print("https://" + most_joyful.strip())

