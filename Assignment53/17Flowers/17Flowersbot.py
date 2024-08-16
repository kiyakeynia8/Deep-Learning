import telebot
from telebot import types
import tensorflow as tf
import cv2
import numpy as np

bot = telebot.TeleBot("7399802797:AAH3yGmxAx3cjftQMjpnF11RSLmlv12DYBg", parse_mode=None)
flowers_name = ['bluebell', 'buttercup', 'coltsfoot', 'cowslip', 'crocus', 'daffodil','daisy', 'dandelion', 'fritillary', 'iris', 'lilyvalley', 'pansy', 'snowdrop', 'sunflower', 'tigerlily', 'tulip', 'windflower']

@bot.message_handler(commands=['start'])
def fname(message):
	welcome = ( "Hello "+ message.from_user.first_name+" welcome to  my Bot :) ")
	markup = types.ReplyKeyboardMarkup(row_width=1)
	itembtn1 = types.KeyboardButton('/help')
	markup.add(itembtn1)
	bot.send_message(message.chat.id, welcome, reply_markup = markup)

@bot.message_handler(commands=['help'])
def fname(message):
	bot.send_message(message.chat.id,"To use this bot, just send a photo of one of the following flowers so that the algorithm can recognize the type of this flower.")
	p = open("17 Flowers/Flowers.jpg", "rb")
	bot.send_photo(message.chat.id, p, None)

@bot.message_handler(content_types = ['photo'])
def fname(message):
    bot.send_message(message.chat.id, 'Please Wait')

    file = message.photo[-1].file_id
    file_info = bot.get_file(file)
    downloaded_file = bot.download_file(file_info.file_path)

    with open("17 Flowers/flower.jpg", 'wb') as new_file:
        new_file.write(downloaded_file)

    img = cv2.imread("17 Flowers/flower.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img ,(224,224))
    img = img / 255
    img = img.reshape(1,224,224,3)
    model = tf.keras.models.load_model("17 Flowers/17Flowers.h5")
    result = np.argmax(model.predict(img))

    bot.send_message(message.chat.id,f'It is a {flowers_name[result]}')

bot.infinity_polling()