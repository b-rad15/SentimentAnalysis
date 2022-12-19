# Article Sentiment Analyzer

Article Sentiment Analyzer is a python program using tensorflow, keras, Natural Language Tool Kit (NLTK), and numpy to estimate the emotions present in an article. It breaks the articles down into their component sentences, then runs them through a deep learning model to predict the emotion present in each sentence, sums those emotions to determine the overall sentiment of the article, then normalizes the sentiment vector so that these results can be compared to other articles in the 6 dimensional emotion space (Anger, Fear, Sadness, Joy, Surprise, Love).

It provides 5 files to serve different purposes:

* [main.py](main.py) is used to train the model
  * model is saved to the folder and files specified on [line 63](main.py#L63)
  * the word vectors used are common crawls 300 dimension, 2 Million word vectors, this can be edited on [line 47](main.py#L47)
  * Train, val, and test (only used for making tokenizer) data is specified on [lines 18-20](main.py#L18-20) 
    * if changing training classes, [line 21](main.py#L21) must also be changed
  * If model is not to be used with an NVIDIA GPU, `cuda_optimized_gru` should be set to `False` on [line 47](main.py#L47)  
    * Note: Model runs and trains roughly 10x faster on NVIDIA GPUs with cuda enabled
  * If the machine has too little memory (System or GPU) to train the model, reduce the `batch_size` on [line 50](main.py#L50) and `gru_output_size` on [helper.py line 64](helper.py#L64)
* [cli.py](cli.py) is used to estimate the emotion of individual statements, it really only exists for testing purposes
  * [line 14](cli.py#L14) is where the .yaml file containing the configuration created in [main.py](main.py#L63) should be specified
  * when run, the file will continually ask for sentences and print their emotion amounts until `quit` is entered
* [predict_all_urls.py](predict_all_urls.py) grabs the most recent reviews from [ign.com](https://www.ign.com/reviews) and estimates the sentimens for all of them, it then outputs the most joyful one
  * It also stores the article urls to articles.txt to speed up repeated runs
  * The number of articles to get should be specified on lines [131](predict_all_urls.py#L131) and [134](predict_all_urls.py#L134)
    * Note: this number will be lower if the site errors and returns fewer than that number of articles
* [analyze_article_cli.py](analyze_article_cli.py) is a command line interface like [cli.py](cli.py) but will take in article links instead of statements
  * like in [cli.py](cli.py) the model file should be specified on [line 11](analyze_article_cli.py#L11)
  * Only [ign](https://www.ign.com) articles are supported at this time but this is only due to the web scraper, it could be expanded to work with other sites
  * [avatar-the-way-of-water-review.svg](avatar-the-way-of-water-review.svg) and [star-wars-knights-of-the-old-republic-review-2.svg](star-wars-knights-of-the-old-republic-review-2.svg) are included as sample chart outputs from this
* [test.py](test.py) is used to anlyze the performance of the model
  * [line 21](test.py#L21) should be modified to the model path
  * [line 19](test.py#L19) should be modified to the test data path

There are also 2 helper files:

* [helper.py](helper.py) contains a number of functions that are used by multiple files
* [ModelProcessing.py](ModelProcessing.py) is a class to store the model, tokenizer, and other necessary data and has functions to read and write them to files for later use

`model-cuda.yaml` and its realted files are contained in this repo so that the user can simply test it without having to train a model themselves. Note that this model was trained on an RTX2070 Super and may use more memory than some GPUs have and will likely run slowly on a CPU only. On my machine it runs at a rate of 608 predictions per second and an accuracy of 92.85%