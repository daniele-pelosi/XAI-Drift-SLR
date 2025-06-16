import string


def calculate_words_scores(tokens_scores, tokenized_text_to_explain):
    tokens_scores.pop(0)
    words_scores = []

    index = -1
    for token_score, token in zip(tokens_scores, tokenized_text_to_explain):
        # token_score is a tuple that contains the string for the token (position 0) and the score (position 1)
        if "Ġ" in token or index == -1:
            index += 1
            words_scores.append(create_word_score(token_score[0], token_score))
        else:
            if token not in string.punctuation:
                words_scores[index] = modify_word_score(token, token_score, words_scores[index])
            else:
                index += 1
                words_scores.append(create_word_score(token, token_score))

    return words_scores


def calculate_words_scores_distil(tokens_scores, tokenized_text_to_explain):
    tokens_scores.pop(0)
    words_scores = []

    index = -1
    for token_score, token in zip(tokens_scores, tokenized_text_to_explain):
        if "##" not in token or index == -1:  # If it's the first index or there is no ## in the token, then it's a new word.
            index += 1
            # token_score is a tuple that contains the string for the token (position 0) and the score (position 1)
            words_scores.append(create_word_score(token_score[0], token_score))
        else:
            token = token.replace("##", "")
            if token not in string.punctuation:
                words_scores[index] = modify_word_score(token, token_score, words_scores[index])
            else:
                index += 1
                words_scores.append(create_word_score(token, token_score))

    return words_scores


def create_word_score(token, token_score):
    return {
                "complete_word": token,
                "scores": [token_score[1]],
                "tokens": [token]
            }


def modify_word_score(token, token_score, word_score_to_update):
    word_score_to_update["complete_word"] += token
    word_score_to_update["scores"].append(token_score[1])
    word_score_to_update["tokens"].append(token)

    return word_score_to_update


