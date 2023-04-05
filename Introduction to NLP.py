import tensorflow as tf

import pandas as pd

#%%
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_df_shuffled = train_df.sample(frac=1, random_state=42)

import random

random_index = random.randint(0, len(train_df) - 5)
for row in train_df[["text", "target"]][random_index:random_index + 5].itertuples():
    _, text, target = row
    print(f"Target: {target}", "(real disaster)" if target > 0 else "(not disaster)")
    print(f"Text:\n{text}\n")
    print("---\n")

from sklearn.model_selection import train_test_split

train_sentence, val_sentence, train_label, val_label = train_test_split(
    train_df_shuffled["text"].to_numpy(),
    train_df_shuffled["target"].to_numpy(),
    test_size=0.1,
    random_state=42
)

import tensorflow as tf

text_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=10000,  # how many words in the dataset
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    ngrams=None,
    output_mode="int",
    output_sequence_length=None,
    pad_to_max_tokens=True,
)

max_vocab_lenght = 10000  # max number of words to have in our vocabulary
max_length = 15  # max length our sequences will be

text_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=max_vocab_lenght,
    output_mode="int",
    output_sequence_length=max_length
)

text_vectorizer.adapt(train_sentence)

embedding = tf.keras.layers.Embedding(
    input_dim=max_vocab_lenght,  # the number of words you have, set input shape
    output_dim=128,  # Because we use GPY, use number of multiple of 8, length of the vector
    input_length=max_length  # How long is each input
)

random_sentence = random.choice(train_sentence)
embedded_sentence = embedding(text_vectorizer([random_sentence]))
print(embedded_sentence)

# Building the baseline model using sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Create the tokenization and modelling pipeline
model_0 = Pipeline([
    ("tfidf", TfidfVectorizer()),  # convert words to number using tfidf
    ("clf", MultinomialNB()),  # Model the text
])

# fit the pipeline the training data
model_0.fit(train_sentence, train_label)

# Evaluate baseline model
baseline_score = model_0.score(val_sentence, val_label)
print(baseline_score)

# Make prediction for the baseline model
baseline_prediction = model_0.predict(val_sentence)
print(baseline_prediction)

# Let us make a function to evaluate our model result
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def calculate_results(y_true, y_pred):
    """
    Calculate model accuracy, precision, recall, and fi_score of a binary
    classification model
    :param y_true: The true label of your validation / test dataset
    :param y_pred: The result of your model prediction
    :return: Accuracy score, precision score, recall score, f1 score
    """
    # Calculate model accuracy
    model_accuracy = accuracy_score(y_true, y_pred)
    model_precision, model_recall, model_f1_score, _ = precision_recall_fscore_support(y_true, y_pred,
                                                                                       average="weighted")
    model_results = {
        "accuracy": model_accuracy,
        "precision": model_precision,
        "recall": model_recall,
        "f1": model_f1_score
    }
    return model_results


baseline_result = calculate_results(y_true=val_label, y_pred=baseline_prediction)
print(baseline_result)

# --------------------------------------------- Model 1 -----------------------------------------
from keras import layers

inputs = layers.Input(shape=(1,), dtype="string", name="input_layer")  # inputs are 1-dimensional
x = text_vectorizer(inputs)  # turn the input text to numbers
x = embedding(x)  # Create an embedding of the numberize inputs
x = layers.GlobalAveragePooling1D()(x)  # Condense the feature vector for each token to one vector
outputs = layers.Dense(1, activation="sigmoid")(x)  # Because it is binary classification
model_1 = tf.keras.Model(inputs, outputs)

model_1.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
)

history_1 = model_1.fit(
    x=train_sentence,
    y=train_label,
    epochs=5,
    validation_data=(val_sentence, val_label)
)

model_1_predictions = model_1.predict(val_sentence)
model_1_predictions = tf.squeeze(model_1_predictions)
model_1_predictions = tf.round(model_1_predictions)
model_1_result = calculate_results(val_label, model_1_predictions)
print(model_1_result)

import numpy as np

print(np.array(list(model_1_result.values())) > np.array(list(baseline_result.values())))

embed_weights = model_1.get_layer("embedding").get_weights()[0]
print(embed_weights.shape)

# --------------------------------------------- Model 2 -----------------------------------------
from keras import layers

inputs = layers.Input(shape=(1,), dtype=tf.string)
x = text_vectorizer(inputs)
x = embedding(x)
x = layers.LSTM(units=64, return_sequences=True)(x)
x = layers.LSTM(units=64)(x)
x = layers.Dense(units=64, activation="relu")(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model_2 = tf.keras.Model(inputs, outputs)

model_2.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)
history_2 = model_2.fit(
    train_sentence,
    train_label,
    epochs=5,
    validation_data=(val_sentence, val_label)
)

model_2_predictions = model_2.predict(val_sentence)
model_2_predictions = tf.squeeze(model_2_predictions)
model_2_predictions = tf.round(model_2_predictions)
model_2_results = calculate_results(y_true=val_label, y_pred=model_2_predictions)
print(model_2_results)

# --------------------------------------------- Model 3 -----------------------------------------
from keras import layers

inputs = layers.Input(shape=(1,), dtype=tf.string)
x = text_vectorizer(inputs)
x = embedding(x)
x = layers.GRU(units=64)(x)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model_3 = tf.keras.Model(inputs, outputs)

model_3.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer="adam",
    metrics=["accuracy"]
)

history_3 = model_3.fit(
    train_sentence,
    train_label,
    epochs=5,
    validation_data=(val_sentence, val_label)
)

model_3_predictions = model_3.predict(val_sentence)
model_3_predictions = tf.squeeze(model_3_predictions)
model_3_predictions = tf.round(model_3_predictions)
model_3_results = calculate_results(y_true=val_label, y_pred=model_3_predictions)
print(model_3_results)
# --------------------------------------------- Model 4 -----------------------------------------
from keras import layers

inputs = layers.Input(shape=(1,), dtype=tf.string)
x = text_vectorizer(inputs)
x = embedding(x)
x = layers.Bidirectional(layer=layers.LSTM(64, return_sequences=True))(x)
x = layers.Bidirectional(layer=layers.GRU(64))(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model_4 = tf.keras.Model(inputs, outputs)

model_4.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
)

history_4 = model_4.fit(
    train_sentence,
    train_label,
    epochs=5,
    validation_data=(val_sentence, val_label)
)

model_4_predictions = model_4.predict(val_sentence)
model_4_predictions = tf.squeeze(model_4_predictions)
model_4_predictions = tf.round(model_4_predictions)
model_4_results = calculate_results(val_label, model_4_predictions)
print(model_4_results)
# --------------------------------------------- Model 5 -----------------------------------------
from keras import layers

inputs = layers.Input(shape=(1,), dtype=tf.string)
x = text_vectorizer(inputs)
x = embedding(x)
x = layers.Conv1D(filters=64, kernel_size=5, activation="relu")(x)
x = layers.GlobalMaxPool1D()(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model_5 = tf.keras.Model(inputs, outputs)

model_5.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer="adam",
    metrics=["accuracy"]
)

history_5 = model_5.fit(
    train_sentence,
    train_label,
    epochs=5,
    validation_data=(val_sentence, val_label)
)

model_5_predictions = model_5.predict(val_sentence)
model_5_predictions = tf.squeeze(model_5_predictions)
model_5_predictions = tf.round(model_5_predictions)
model_5_results = calculate_results(y_true=val_label, y_pred=model_5_predictions)
print(model_5_results)

import numpy as np

print(np.array(list(model_5_results)) > np.array(list(baseline_result)))
# --------------------------------------------- Model 6 -----------------------------------------
import tensorflow_hub as hub

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
embed_sample = embed(["When you are wake up, just go to bed again"])
print(embed_sample)

sentence_encoder_layer = hub.KerasLayer(
    "https://tfhub.dev/google/universal-sentence-encoder/4",
    input_shape=[],
    dtype=tf.string,
    trainable=False
)

# Create the model using sequential API
model_6 = tf.keras.Sequential([
    sentence_encoder_layer,  # The layer handle everything (vectorization, embedding)
    layers.Dense(1, activation="sigmoid")  # output layer
])

model_6.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
)

history_6 = model_6.fit(
    train_sentence,
    train_label,
    epochs=5,
    validation_data=(val_sentence, val_label)
)

model_6_predictin = model_6.predict(val_sentence)
model_6_predictin = tf.squeeze(model_6_predictin)
model_6_predictin = tf.round(model_6_predictin)
model_6_results = calculate_results(y_true=val_label, y_pred=model_6_predictin)
print(model_6_results)

print(model_6_results.values())
print(baseline_result.values())

print(np.array(list(model_6_results)) > np.array(list(baseline_result)))
comparing_list = np.array(list(model_6_results.values())) > np.array(list(baseline_result.values()))
print(comparing_list)

# Adding Dense layer to model 6 and evaluate it
model_6_updated = tf.keras.Sequential([
    sentence_encoder_layer,  # The layer handle everything (vectorization, embedding)
    layers.Dense(units=64, activation="relu"),
    layers.Dense(1, activation="sigmoid")  # output layer
])

model_6_updated.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
)

history_6_updated = model_6_updated.fit(
    train_sentence,
    train_label,
    epochs=5,
    validation_data=(val_sentence, val_label)
)

model_6_updated_predictions = model_6_updated.predict(val_sentence)
model_6_updated_predictions = tf.squeeze(model_6_updated_predictions)
model_6_updated_predictions = tf.round(model_6_updated_predictions)
model_6_updated_results = calculate_results(y_true=val_label, y_pred=model_6_updated_predictions)
print(model_6_updated_results)

comparing_list = np.array(list(model_6_updated_results.values())) > np.array(list(baseline_result.values()))
print(comparing_list)
# --------------------------------------------- Model 7 -----------------------------------------
train_10_percent = train_df_shuffled[["text", "target"]].sample(frac=0.1, random_state=42)
train_sentence_10_percent = train_10_percent['text'].to_list()
train_label_10_percent = train_10_percent['target'].to_list()

model_7 = tf.keras.models.clone_model(model_6)  # because same architecture of model 6
model_7.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
)
history_7 = model_7.fit(
    train_sentence_10_percent,
    train_label_10_percent,
    epochs=5,
    validation_data=(val_sentence, val_label)
)

# Fixing the leakage of the data
train_10_percent_split = int(0.1 * len(train_sentence))
train_sentence_10_percent = train_sentence[:train_10_percent_split]
train_label_10_percent = train_label[:train_10_percent_split]

model_7.fit(
    train_sentence_10_percent,
    train_label_10_percent,
    epochs=5,
    validation_data=(val_sentence, val_label)
)

# --------------------------------------------- Comparing  -----------------------------------------
all_models_resutl = pd.DataFrame({
    "baseline": baseline_result.values(),
    "model_1": model_1_result.values(),
    "model_2": model_2_results.values(),
    "model_3": model_3_results.values(),
    "model_4": model_4_results.values(),
    "model_5": model_5_results.values(),
    "model_6": model_6_results.values(),
    "model_6_updated": model_6_updated_results.values()
}, index=["accuracy", "precision", "recall", "f1"])

all_models_resutl = all_models_resutl.transpose()
import matplotlib.pyplot as plt

all_models_resutl.plot(kind="bar", figsize=(10, 7)).legend(bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()
plt.show()

all_models_resutl.sort_values("accuracy", ascending=False)["accuracy"].plot(kind="bar")
plt.tight_layout()
plt.show()

# Saving the model
model_6.save("model_6.h5")

# Load the model
loaded_model = tf.keras.models.load_model("model_6.h5", custom_objects={"KerasLayer": hub.KerasLayer})

model_6.save("model_6")
loaded_model = tf.keras.models.load_model("model_6")
print(loaded_model.evaluate(val_sentence, val_label))

# -------------------------------- Fixing the wrong in the model-----------------
import wget, zipfile

wget.download(url="https://storage.googleapis.com/ztm_tf_course/08_model_6_USE_feature_extractor.zip")
ref = zipfile.ZipFile("08_model_6_USE_feature_extractor.zip", "r")
ref.extractall()
ref.close()

model_6_pretrained = tf.keras.models.load_model("08_model_6_USE_feature_extractor")
print(model_6_pretrained.evaluate(val_sentence, val_label))

model_6_pretrained_probs = model_6_pretrained.predict(val_sentence)
model_6_pretrained_predictions = tf.squeeze(model_6_pretrained_probs)
model_6_pretrained_predictions = tf.round(model_6_pretrained_predictions)
model_6_pretrained_results = calculate_results(val_label, model_6_pretrained_predictions)
print(model_6_pretrained_results)

model_6_pretrained_dataframe = pd.DataFrame({
    "Text": val_sentence,
    "Target": val_label,
    "Prediction": model_6_pretrained_predictions,
    "Probs": tf.squeeze(model_6_pretrained_probs)
})

most_wrong = model_6_pretrained_dataframe[
    model_6_pretrained_dataframe["Target"] != model_6_pretrained_dataframe["Prediction"]]
print(len(most_wrong))

import time
def pred_timer(model, samples):
    """
    This function is timing how long a model takes to make a prediction on samples
    :param model: The model you want to mak predict
    :param samples: Some sample to make prediction on them
    :return: the time of prediction
    """
    start_time = time.perf_counter()  # Get start time
    model.predict(samples)  # make prediction
    end_time = time.perf_counter()  # Get finish time
    total_time = end_time - start_time  # calculate how long predictions took to make
    time_per_pred = total_time / len(samples)
    return total_time,

baseline_time = pred_timer(model_0, val_sentence)
model_1_time = pred_timer(model_1, val_sentence)
model_2_time = pred_timer(model_2, val_sentence)
model_3_time = pred_timer(model_3, val_sentence)
model_4_time = pred_timer(model_4, val_sentence)
model_5_time = pred_timer(model_5, val_sentence)
model_6_time = pred_timer(model_6, val_sentence)
model_6_updated_time = pred_timer(model_6_updated, val_sentence)



baseline_accuracy = accuracy_score(val_label ,baseline_prediction)
model_1_accuracy = accuracy_score(val_label, model_1_predictions)
model_2_accuracy = accuracy_score(val_label, model_2_predictions)
model_3_accuracy = accuracy_score(val_label, model_3_predictions)
model_4_accuracy = accuracy_score(val_label, model_4_predictions)
model_5_accuracy = accuracy_score(val_label, model_5_predictions)
model_6_accuracy = accuracy_score(val_label, model_6_predictin)
model_6_updated_accuracy = accuracy_score(val_label, model_6_updated_predictions)


all_models_time = pd.DataFrame({
    "Time":[baseline_time, model_1_time, model_2_time, model_3_time, model_4_time, model_5_time, model_6_time, model_6_updated_time],
    "Accuracy":[baseline_accuracy, model_1_accuracy, model_2_accuracy, model_3_accuracy, model_4_accuracy, model_5_accuracy, model_6_accuracy, model_6_updated_accuracy]
}, index=["Baseline", "Model_1", "Model_2", "Model_3", "Model_4", "Model_5", "Model_6", "Model_6_updated"])

plt.scatter(all_models_time["Time"], all_models_time["Accuracy"])
plt.show()