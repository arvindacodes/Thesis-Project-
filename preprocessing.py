file = open('/Users/aravi/PycharmProjects/final_thesis_model/dataset/chat_test.txt', 'r', encoding="utf8") ## original twitter dataset
input_file= file.read()
# print(input_file)
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"
                           u"\U0001F300-\U0001F5FF"
                           u"\U0001F680-\U0001F6FF"
                           u"\U0001F1E0-\U0001F1FF"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u"\U00010000-\U0010ffff"
                           u"\u200d"
                           u"\u2640-\u2642"
                           "]", flags=re.UNICODE)
input_file = emoji_pattern.sub(r'', input_file)


"""  Removing URL in the 2nd step"""
url_regex = re.compile(r"(http|https|ftp)://[a-zA-Z0-9\./]+")
input_file = re.sub(url_regex, '', input_file)


"""removing characters [\], ['] and ["] from text. this also removes "'" in words like "didn't" """
input_file = re.sub(r"\\", "", input_file)
input_file = re.sub(r"\'", "", input_file)
input_file = re.sub(r"\"", "", input_file)


""" removing spl characters from text"""
input_file = re.sub('[^\w\s]', '' , input_file)
input_file = re.sub('_','', input_file)



# # replacing punctuation characters with spaces
# filter = '!"\'โ#$%&()*+,-./:;<=>?@[\\]^_`{|}~คธ\t\n'
# translate_dict = dict((c, " ") for c in filter)
# translate_map = str.maketrans(translate_dict)
# input_file = input_file.translate(translate_map)


#removing all numbers by replacing them with ' '

input_file = re.sub(r'[0-9]+', '', input_file)

# converting text to lowercase
input_file = input_file.strip().lower()
# # print(input_file)