import pathlib
import pickle
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import spacy
sp = spacy.load('en_core_web_trf')


def generate_global_explanation(interpretability_directory, interpretability_step):
    interpret_results_file = interpretability_directory + "interpret_results_" + str(interpretability_step) + ".pickle"
    interpret_results = pickle.load(open(interpret_results_file, "rb"))

    global_words_score = {}
    for current_interpretability in interpret_results:
        if current_interpretability["corrected_prediction"] and current_interpretability["true_class"]:
            for word_score in current_interpretability["words_scores"]:
                mean_score = np.mean(word_score["scores"])
                word = word_score["complete_word"]

                if word in global_words_score:
                    global_words_score[word].append(mean_score)
                else:
                    global_words_score[word] = [mean_score]

    global_words_score_mean = {}
    for word in global_words_score:
        global_words_score_mean[word] = np.mean(global_words_score[word])

    sorted_scores = dict(sorted(global_words_score_mean.items(), key=lambda item: item[1], reverse=True)[:20])
    print(sorted_scores)
    generate_word_cloud(sorted_scores)
    print("Global")


def generate_global_chart(interpretability_directory, interpretability_step, model_name):
    base_directory = str(pathlib.Path(__file__).parent)

    interpret_results_file = base_directory + interpretability_directory + "interpret_results_" + str(interpretability_step) + ".pickle"
    interpret_results = pickle.load(open(interpret_results_file, "rb"))

    all_stop_words = sp.Defaults.stop_words

    global_words_score = {}
    for current_interpretability in interpret_results:
        # current_interpretability["corrected_prediction"] and  Now we are considering even when the model classifies with a mistake. We want to see all values.
        if current_interpretability["true_class"]:
            for word_score in current_interpretability["words_scores"]:
                word = word_score["complete_word"]
                if word not in all_stop_words:
                    mean_score = np.sum(word_score["scores"])  # We sum the score of each token that belongs to a word.

                    if word in global_words_score:
                        global_words_score[word].append(mean_score)
                    else:
                        global_words_score[word] = [mean_score]

    global_words_score_relative = {}
    global_words_score_absolute = {}
    global_words_num_present = {}
    for word in global_words_score:
        global_words_score_relative[word] = np.mean(global_words_score[word])
        global_words_score_absolute[word] = np.sum(global_words_score[word]) # / len(interpret_results)
        global_words_num_present[word] = len(global_words_score[word])

    sorted_scores_relative = dict(sorted(global_words_score_relative.items(), key=lambda item: item[1], reverse=True)[:20])
    sorted_scores_absolute = dict(sorted(global_words_score_absolute.items(), key=lambda item: item[1], reverse=True)[:20])

    absolute_words_frequencies = {}
    for word_key in sorted_scores_absolute.keys():
        absolute_words_frequencies[word_key] = global_words_num_present[word_key]

    relative_words_frequencies = {}
    for word_key in sorted_scores_relative.keys():
        relative_words_frequencies[word_key] = global_words_num_present[word_key]

    sorted_absolute_words_frequencies = dict(sorted(absolute_words_frequencies.items(), key=lambda item: item[1], reverse=True)[:20])
    sorted_relative_words_frequencies = dict(sorted(relative_words_frequencies.items(), key=lambda item: item[1], reverse=True)[:20])

    generate_bar_chart(sorted_scores_relative, relative_words_frequencies, model_name + " - Relative Relevance Score", "Relevance Score")
    generate_bar_chart(sorted_scores_absolute, absolute_words_frequencies, model_name + " - Global Sum of Relevance Score", "Sum of Relevance Score")
    # generate_bar_chart(sorted_words_num_present)


def generate_global_interpretability_multi(interpretability_directory, interpretability_step, model_name, number_of_labels=6,
                                           labels_descriptions=[]):
    base_directory = str(pathlib.Path(__file__).parent)

    interpret_results_file = base_directory + interpretability_directory + "interpret_results_" + str(interpretability_step) + ".pickle"
    all_interpret_results = pickle.load(open(interpret_results_file, "rb"))

    all_stop_words = sp.Defaults.stop_words

    for label in range(number_of_labels):
        interpret_results = all_interpret_results[str(label)]
        global_words_score = {}
        for current_interpretability in interpret_results:
            # current_interpretability["corrected_prediction"] and  Now we are considering even when the model classifies with a mistake. We want to see all values.
            if current_interpretability["true_value"] or not current_interpretability["true_value"]:
                for word_score in current_interpretability["words_scores"]:
                    word = word_score["complete_word"]
                    if word not in all_stop_words:
                        mean_score = np.sum(word_score["scores"])  # We sum the score of each token that belongs to a word.

                        if word in global_words_score:
                            global_words_score[word].append(mean_score)
                        else:
                            global_words_score[word] = [mean_score]

        global_words_score_relative = {}
        global_words_score_absolute = {}
        global_words_num_present = {}
        for word in global_words_score:
            global_words_score_relative[word] = np.mean(global_words_score[word])
            global_words_score_absolute[word] = np.sum(global_words_score[word])  # / len(interpret_results)
            global_words_num_present[word] = len(global_words_score[word])

        sorted_scores_relative = dict(sorted(global_words_score_relative.items(), key=lambda item: item[1], reverse=True)[:20])
        sorted_scores_absolute = dict(sorted(global_words_score_absolute.items(), key=lambda item: item[1], reverse=True)[:20])

        absolute_words_frequencies = {}
        for word_key in sorted_scores_absolute.keys():
            absolute_words_frequencies[word_key] = global_words_num_present[word_key]

        relative_words_frequencies = {}
        for word_key in sorted_scores_relative.keys():
            relative_words_frequencies[word_key] = global_words_num_present[word_key]

        sorted_absolute_words_frequencies = dict(sorted(absolute_words_frequencies.items(), key=lambda item: item[1], reverse=True)[:20])
        sorted_relative_words_frequencies = dict(sorted(relative_words_frequencies.items(), key=lambda item: item[1], reverse=True)[:20])

        generate_bar_chart(sorted_scores_relative, relative_words_frequencies,
                           model_name + " " + labels_descriptions[label] + " - Relative Relevance Score", "Relevance Score")
        generate_bar_chart(sorted_scores_absolute, absolute_words_frequencies,
                           model_name + " " + labels_descriptions[label] + " - Global Sum of Relevance Score", "Sum of Relevance Score")


def generate_word_cloud(words_scores):
    wordcloud = WordCloud(width=400, height=400,
                          background_color='white',
                          min_font_size=5, collocations=False).generate_from_frequencies(words_scores)

    # plot the WordCloud image
    plt.figure(figsize=(4, 4), facecolor=None)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.show()


def generate_bar_chart(words_scores, words_frequencies, chart_title, x_label=""):
    words = words_scores.keys()
    scores = words_scores.values()

    plt.rcdefaults()
    fig, ax = plt.subplots()

    word_position = list(range(0, 4 * len(words), 4))

    ax.barh(word_position, scores, height=1.5, align="center", color="#000066")
    ax.set_yticks(word_position, labels=words)
    ax.invert_yaxis()
    ax.set_title(chart_title, fontsize=10)
    plt.xlabel(x_label)

    frequency_values = list(words_frequencies.values())
    test = list(scores)
    for index in range(len(words)):
        plt.text(test[index], word_position[index], frequency_values[index], ha="left", fontsize=7,
                 bbox=dict(facecolor="#ffb836", alpha=.3))

    relevance_score_path = mpatches.Patch(facecolor='#000066', label='Relevance Score')
    word_frequency_patch = mpatches.Patch(facecolor='#ffb836', alpha=.3, label='Word Frequency')
    ax.legend(handles=[relevance_score_path, word_frequency_patch], loc="lower right")

    fig.set_figheight(6)
    fig.set_figwidth(5)
    fig.tight_layout(pad=0)

    plt.show()
