
from kaggle_secrets import UserSecretsClient
YANDEX_API_KEY = UserSecretsClient().get_secret("YANDEX_API_KEY")
import requests
def translate(x, key, src='ru', dest='en'):
    original = x.unique()
    url = 'https://translate.yandex.net/api/v1.5/tr.json/translate'
    params = dict(
        key=key,
        lang=src+'-'+dest
    )
    payload = {'text': original}
    response = requests.post(url=url, params=params, data=payload)
    translated_text = response.json()['text']
    dictionary = dict(zip(original, translated_text))
    return([dictionary.get(item, item) for item in x])
import pandas as pd
categories = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
categories['item_category_name_en'] = translate(categories['item_category_name'], YANDEX_API_KEY)
categories.sample(10)
