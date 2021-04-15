# %%
## Import libraries ----
import numpy as np
import pandas as pd  # Requires version 1.2.0 or later (because of the use of merge(how='cross')
from pandas.core.common import flatten
from annoy import AnnoyIndex
from gensim.models import Word2Vec
from efficient_apriori import apriori

import nltk
nltk.download('wordnet')

from tqdm import tqdm
import zipfile as zp
from art import *
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
import plotly.express as px

# from PyDictionary import PyDictionary 
import random
import time

# %%
## Constants ----
# Limit of number of orders to process
ORDERS_SPEED_LIMIT = 10000
# Color constants for the console
COLOR_CONSTANT = {'input': '\033[94m', 'warning': '\033[93m', 'error': '\033[91m', 'note': '\033[96m', 'end': '\033[0m'}
# Number of orders/baskets to pull similar to the requested
NUMBER_OF_RETURNS = 10
# Number of dimensions of the vector annoy is going to store. 
VECTOR_SIZE = 20
# Number of trees for queries. When making a query the more trees the easier it is to go down the right path. 
TREE_QUERIES = 10
# Number of product recommendation as maximum
NUMBER_OUTPUT_PRODUCTS = 10
# Sample size for the TSNE model and plot
TSNE_SAMPLE_SIZE = 1000
# Threshold for a minimum support
THRESHOLD_SUPPORT = 1e-3
# Threshold for the maximun number of products to bring
THRESHOLD_TOP = 10
# Threshold for distance, based on the quantile calculation of the basket distances
THRESHOLD_DISTANCE = 0.1

# %%
## Read data ----
# Data column description
# https://gist.github.com/jeremystan/c3b39d947d9b88b3ccff3147dbcf6c6b

zf1 = zp.ZipFile("Data/order_products__prior.csv.zip")
order_products_prior = pd.read_csv(zf1.open('order_products__prior.csv'))

zf1 = zp.ZipFile("Data/orders.csv.zip")
orders = pd.read_csv(zf1.open('orders.csv'))

zf1 = zp.ZipFile("Data/products.csv.zip")
products = pd.read_csv(zf1.open('products.csv'))

zf1 = zp.ZipFile("Data/departments.csv.zip")
departments = pd.read_csv(zf1.open('departments.csv'))

zf1 = zp.ZipFile("Data/aisles.csv.zip")
aisles = pd.read_csv(zf1.open('aisles.csv'))

# %%
### Synonyms ----
# Manually extracted from https://www.thesaurus.com/browse/synonym
# departments_synonyms = pd.read_csv('Data/departments synonyms.csv')

# Synonyms for all the aisle using the library `PyDictionary`
# This idea is too slow to implement
# # https://stackoverflow.com/a/36827814/3780957
# dictionary=PyDictionary() 
# aisles['split'] = aisles['aisle'].str.split()
# b = aisles.explode('split')
# c = b['split'].apply(dictionary.synonym)

# %%
### Merge data ----

# Make everything lowercase.
products['products_mod'] = products['product_name'].str.lower()
# Clean special characters.
products['products_mod'] = products['products_mod'].str.replace('\W', ' ', regex=True)
# Split products into terms: Tokenize.
products['products_mod'] = products['products_mod'].str.split()

# # Merge the synonyms perse
# departments_synonyms = departments_synonyms.groupby('department')['synonyms'].apply(list)
# departments_synonyms = pd.merge(departments, departments_synonyms, on="department", how='outer').fillna('')

# Merge the department and aisle names into the dataframe. 
products = pd.merge(products, departments, on="department_id", how='outer')
products = pd.merge(products, aisles, on="aisle_id", how='outer')

# https://stackoverflow.com/a/43898233/3780957
# https://stackoverflow.com/a/57225427/3780957
# Remove synonyms here in the list
products['products_mod'] = products[['products_mod', 'aisle', 'department']].values.tolist()
products['products_mod'] = products['products_mod'].apply(lambda x:list(flatten(x)))

# %%
# Steam and lemmatisation of the product name
# https://stackoverflow.com/a/24663617/3780957
# https://stackoverflow.com/a/25082458/3780957
# https://en.wikipedia.org/wiki/Lemmatisation

lemma = nltk.wordnet.WordNetLemmatizer()
sno = nltk.stem.SnowballStemmer('english')
products['products_lemma'] = products['products_mod'].apply(lambda row:[lemma.lemmatize(item) for item in row])
products['products_lemma'] = products['products_lemma'].apply(lambda row:[sno.stem(item) for item in row])

# %%
## EDA ----

def eda(order_baskets, orders_filter, products):
    ### Histogram ----
    _, axs = plt.subplots(2, 1, constrained_layout=True, sharex=False, sharey=False, figsize=(15,15))

    x = order_baskets.apply(len)
    axs[0].set_title('Number of items per order')
    axs[0].hist(x, bins=30, color='c', edgecolor='k', alpha=0.65)
    axs[0].set(xlabel="Items per order", ylabel="Number of orders")
    axs[0].axvline(x.mean(), color='k', linestyle='dashed', linewidth=1)
    # # _, max_ylim = axs[0].ylim()
    max_ylim = 1500  # Fixed value
    axs[0].text(x.mean()*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(x.mean()))

    ### Departments and Aisles ----
    order_products = pd.merge(orders_filter, products, on="product_id", how='left')
    x = order_products['department'].value_counts()
    axs[1].set_title('Most popular Departments')
    axs[1].bar(x.index, x, color='c', edgecolor='k', alpha=0.65)
    axs[1].set(xlabel=None, ylabel="Number of orders")
    axs[1].tick_params(axis='x', labelrotation=90)
    axs[1].axhline(x.mean(), color='k', linestyle='dashed', linewidth=1)
    axs[1].text(10, x.mean()*1.1, 'Mean: {:.2f}'.format(x.mean()))

    plt.show(block=True)

# %%
## `Word2Vec` model ----
### Train the model ----
# The `products_lemma` column is ready to be used as an input for the `Word2Vec` model. 

# to define the maximun window
window_max = max(products['products_lemma'].apply(lambda x:len(x)))

# size=20: In order to make `Word2Vec` a little bit quicker and for memory efficiency we're going to use 20 dimensions.
# window=49: In order to make sure all words are used in training the model, we're going to set a large.
# workers=-1: means all the available cores in the CPU.
w2vec_model = Word2Vec(list(products['products_lemma']), size=VECTOR_SIZE, window=window_max, min_count=1, workers=-1)

# %%
### Vector calculation for products ----
# Loop through each product and obtain the average of each string that makes a product. <br>
# For this dictionary we're not gong to store the product name as its key, but the product ID. 

# Cycle through each word in the product name to generate the vector.
prods_w2v = dict()
for row, product in tqdm(products.iterrows()):
    word_vector = list()
    for word in product['products_lemma']:
        word_vector.append(w2vec_model.wv[word])

    prods_w2v[product['product_id']] = np.average(word_vector, axis=0)

# Save vector values in list form to the dataframe.
products['vectors'] = prods_w2v.values()

# %%
## TSNE model plot function ----

def tsne_plot(df, title, color=None, product_flag=False, auto_open=True, sample_size=TSNE_SAMPLE_SIZE):
    # Data sample, to speedup the execution
    df_tsne_data = df.sample(n=sample_size, random_state=42)

    # Train the TSNE MODEL
    tsne_model = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3500, random_state=42)
    new_values = tsne_model.fit_transform(list(df_tsne_data['vectors']))

    # Prepare data
    x = list()
    y = list()
    for i in range(new_values.shape[0]):
        x.append(new_values[i][0])
        y.append(new_values[i][1])

    if color is not None:
        marker_ = dict(color=list(df_tsne_data[color]), colorscale='Viridis', showscale=False)
        if product_flag:
            text_ = df_tsne_data[['product_name', 'aisle', 'department']].agg('<br>'.join, axis=1)
        else:
            text_ = color + ": " +  df_tsne_data[color].astype(str)
    else:
        marker_ = text_ = None
    
    trace = go.Scatter(
        x = x,
        y = y,
        mode = 'markers',
        text = text_,
        hoverinfo = 'text',
        marker = marker_
    )

    layout = go.Layout(
        title = title,
        hovermode = 'closest',
        xaxis = dict(title='Dimension one', autorange=True),
        yaxis = dict(title='Dimension two', autorange=True))

    # Create plot
    fig = go.Figure(data=[trace], layout=layout)

    # https://www.programcreek.com/python/example/103216/plotly.graph_objs.Layout
    pyo.plot(fig, filename=f'Plots/tsne_plot_{title}.html', auto_open=auto_open)

# %%
## TSNE model plot function, with selection ----

def tsne_plot2(df, title, selection, hover=None, auto_open=True, sample_size=TSNE_SAMPLE_SIZE):

    # Data sample, to speedup the execution
    df_tsne_data = df.sample(n=sample_size, random_state=42)
    df_tsne_data['size'] = 1
    df_tsne_data['color'] = 'Others'

    selection = selection.copy()  # To avoid a warning
    selection['size'] = 5
    selection['color'] = 'Selection'

    df_tsne_data = df_tsne_data.append(selection)

    # Train the TSNE MODEL
    tsne_model = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3500, random_state=42)
    tsne_values = tsne_model.fit_transform(list(df_tsne_data['vectors']))

    df_tsne_data['tsne-2d-one'] = tsne_values[:, 0]
    df_tsne_data['tsne-2d-two'] = tsne_values[:, 1]

    if hover is not None:
        df_tsne_data['hover'] = df_tsne_data[hover]
    else:
        df_tsne_data['hover'] = df_tsne_data[['product_name', 'aisle', 'department']].agg('<br>'.join, axis=1)

    df_tsne_data.sort_values(by='color', ascending=False, inplace=True)

    fig = px.scatter(df_tsne_data, x="tsne-2d-one", y="tsne-2d-two",
                    color='color', 
                    size="size", size_max=8,
                    title=title,
                    hover_data=['hover'],
                    labels={
                        "tsne-2d-one": "Dimension one",
                        "tsne-2d-two": "Dimension two",
                        "color": "Color reference"
                    })
    pyo.plot(fig, filename=f'Plots/tsne_plot_2_{title}.html', auto_open=auto_open)

# %%
## Using `annoy` model ----
# Source: https://github.com/spotify/annoy
# About the distance metric: https://en.wikipedia.org/wiki/Euclidean_distance

def annoy_build(df, id, metric='euclidean'):
    m = AnnoyIndex(VECTOR_SIZE, metric=metric) 
    m.set_seed(42)
    for _, row in df.iterrows():
        m.add_item(row[id], row['vectors'])
    m.build(TREE_QUERIES)
    return m

# %%
### Train `annoy` for `product` ----
# We need to specify ahead of time to annoy that there are 20 vector dimensions. Defined as a constant at `VECTOR_SIZE`.
# We also specify we want the model to find distances using `euclidean` distance.

# Specify the metric to be used for computing distances. 
p = annoy_build(products, 'product_id')

# %%
### Train `annoy` for `orders` ----
# In order to obtain the vector for each list we need to import the orders csv. <br>
# The order_products_prior has the order_id and the product_id (this is why we keeping product IDs as a key is useful).
# In order to make sure the calculations are personal computer friendly, we're going to limit the number of orders we are operating on. 

orders_filter = order_products_prior[order_products_prior.order_id < ORDERS_SPEED_LIMIT]
# Alternative method for filering
# mask_orders = orders['order_id'].value_counts()[orders['order_id'].value_counts() >= 50]
# orders_filter = orders[orders['order_id'].isin(mask_orders.keys())]
order_baskets = orders_filter.groupby('order_id')['product_id'].apply(list)

order_w2v = dict()
for index, row in tqdm(order_baskets.items()):
    word_vector = list()
    for item_id in row:
        word_vector.append(p.get_item_vector(item_id))
    order_w2v[index] = np.average(word_vector, axis=0)

df_order_baskets = pd.DataFrame({'order_id': order_baskets.index, 'product_id': order_baskets.values})
df_order_baskets['vectors'] = order_w2v.values()

# Specify the metric to be used for computing distances. 
b = annoy_build(df_order_baskets, 'order_id')

# %%
### Train `annoy` for `user` ----
# Creating an `annoy` object to index the `user` information

user_basket = pd.merge(df_order_baskets, orders, on="order_id", how='inner')
user_basket = user_basket.groupby('user_id').apply(lambda x: [list(x['vectors']), list(x['product_id'])]).apply(pd.Series)
user_basket.columns =['vectors','product_id']
user_basket['vectors'] = user_basket['vectors'].apply(lambda x: tuple(np.average(x, axis=0)))
user_basket['product_id'] = user_basket['product_id'].apply(lambda x: [item for sublist in x for item in sublist])
user_basket['product_id'] = user_basket['product_id'].apply(lambda x: list(set(x)))
df_user_basket = user_basket.reset_index()

# Specify the metric to be used for computing distances. 
u = annoy_build(df_user_basket, 'user_id')

# %%
## Similar ----
# Refer here for all functions: https://github.com/spotify/annoy
### Auxiliary functions ----

# List the unique products maintaining the original order
# https://stackoverflow.com/a/480227/3780957
def unique_preserve_order(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

# Sort recommendations by `lift`, and filter if the products are too close
def product_lift(basket, input = None, order_baskets=order_baskets, th_support=THRESHOLD_SUPPORT, th_n=THRESHOLD_TOP, products=products):
    # Force to include the manual `input`
    recommendations = basket['product_id'].tolist()
    if input is not None:
        recommendations.extend(input)
    recommendations = set(recommendations)

    # Baskets with only the recommended products by the w2v
    order_baskets_ = order_baskets.explode()
    order_baskets_ = order_baskets_[order_baskets_.isin(recommendations)]
    order_baskets_ = order_baskets_.groupby(level=0).apply(list)
    order_baskets_ = order_baskets_.to_list()

    # Calculate `apriori` rules using a efficient library to speed up the calculation
    _, rules = apriori(order_baskets_, min_support=th_support, min_confidence=1e-2, max_length=5)
    
    # Multiple filters, but due to the lack of orders, are limiting the number of results, so a simple filter is active
    if input is not None:
        rules_rhs = filter(lambda rule: \
            # # `input` is the `lhs`, and `input` is not in the `rhs`
            # (all(x in rule.lhs for x in input) and not all(x in rule.rhs for x in input)) or \
            # # `input` is the `lhs`, `lhs` has only 1 element, and `input ` is not in the `rhs`
            # (all(x in input for x in rule.lhs) and len(rule.lhs) == 1 and not all(x in rule.rhs for x in input)) \
            # `input ` is not in the `rhs`
            not all(x in rule.rhs for x in input)
            , rules)
    else:
        rules_rhs = rules

    # Combine all the rules found in the data
    # Sorted by highest lift
    rule_combined = list()
    for rule in sorted(rules_rhs, key=lambda rule: rule.lift, reverse=True):
        # print(rule)
        rule_combined.extend(rule.rhs)

    # List the unique products maintaining the original order
    product_recommendation = unique_preserve_order(rule_combined)

    ## The following code, filters the recommendations after `lift`, based on the distance between the products
    # List of products
    prod = pd.DataFrame({'product_id': product_recommendation})
    prod_cross_join = prod.merge(prod, how='cross')
    # Calculate the distance between all the products
    prod_cross_join['distance'] = prod_cross_join.apply(lambda row: p.get_distance(row['product_id_x'], row['product_id_y']), axis=1)
    # Remove the same product (distance==0)
    prod_cross_join = prod_cross_join[prod_cross_join['distance']!=0]
    prod_cross_join.sort_values('distance', ascending=False)
    # Looking for closest products
    # Threshold for the filter, 10% of the distance (defined at `THRESHOLD_DISTANCE` constant)
    th_distance = np.quantile(prod_cross_join, THRESHOLD_DISTANCE)
    for id in product_recommendation:
        to_be_removed = prod_cross_join.loc[(prod_cross_join['product_id_x']==id) & (prod_cross_join['distance']<th_distance), 'product_id_y']
        prod_cross_join = prod_cross_join[~prod_cross_join['product_id_x'].isin(to_be_removed)]
    # List of final recommendations after the filters and thresholds
    prod_after_filtered = prod_cross_join['product_id_x'].unique()
    # Retain the order from the `lift`
    product_recommendation_filtered = pd.DataFrame({'product_recommendation': product_recommendation}).set_index('product_recommendation').loc[prod_after_filtered].reset_index()
    # Recall the products in the previous order
    product_recommendation_product = products.set_index("product_id").loc[product_recommendation_filtered['product_recommendation']].reset_index()

    return product_recommendation_product[['product_name', 'department', 'aisle']].head(th_n)

# Finds the recommended basket, based on the `Word2Vec` vector as input
def basket_recompose(w2v, b=b, order_baskets=order_baskets):
    # Search for a similar basket in `b`
    similar_baskets = b.get_nns_by_vector(w2v, NUMBER_OF_RETURNS, search_k=-1, include_distances=False)
    basket_recompose = pd.DataFrame({'order_id': similar_baskets, 'product_id': order_baskets[similar_baskets].values}).explode('product_id')

    return basket_recompose

# %%
### Calculate baskets ----
# Based on different inputs, a different method of calculating a basket.

# From a list of products, recommends a basket
def basket_input_product_list(x):
    word_vector = list()
    for item_id in x:
        word_vector.append(p.get_item_vector(item_id))
    product_w2v = np.average(word_vector, axis=0)

    # Search for a similar basket in `b`
    basket = basket_recompose(product_w2v)
    # Remove the manually selected products. Cleanup the output
    basket = basket[~basket['product_id'].isin(x)]
    
    basket_input = products[products['product_id'].isin(x)]
    basket_input_names = basket_input['product_name'].values

    return product_lift(basket, x), basket_input_names, basket_input

# Form a particular user, recommends a basket. Also report the users that are similar to the input.
def basket_input_user(x):
    user_w2v = u.get_item_vector(x)
    selection_w2v = pd.DataFrame({'user_id': x, 'vectors': [tuple(user_w2v),]})

    # Search for similar users in `u`
    similar_users = u.get_nns_by_item(x, NUMBER_OF_RETURNS, search_k=-1, include_distances=False)[1:]

    # Products from the user
    input = df_user_basket.loc[df_user_basket['user_id'] == x, 'product_id'][0]
    products_user_input = products[products['product_id'].isin(input)]
    products_user_input_name = products_user_input['product_name'].tolist()

    # Search for a similar basket in `b`
    basket = basket_recompose(user_w2v)
    return product_lift(basket, input), similar_users, selection_w2v, products_user_input_name

# From a list of users, recommends a basket
def basket_input_user_list(x):
    word_vector = list()
    for item_id in x:
        word_vector.append(tuple(u.get_item_vector(item_id)))
    user_w2v = np.average(word_vector, axis=0)
    # Selected users
    selection_w2v = pd.DataFrame({'user_id': list(x,), 'vectors': list(word_vector,)})

    # Products from the list of users
    input = df_user_basket.loc[df_user_basket['user_id'].isin(x), 'product_id']
    input = [item for sublist in input for item in sublist]
    products_user_input = products[products['product_id'].isin(input)]
    products_user_input_name = products_user_input['product_name'].tolist()

    # Search for a similar basket in `b`
    basket = basket_recompose(user_w2v)
    return product_lift(basket, input), x, selection_w2v, products_user_input_name

# From a particular order, recommends a basket
def basket_input_order(x):
    order_w2v = b.get_item_vector(x)
    selection_w2v = pd.DataFrame({'order_id': x, 'vectors': [tuple(order_w2v),]})

    # Products from the order
    input = df_order_baskets.loc[df_order_baskets['order_id'] == x, 'product_id']
    input = [item for sublist in input for item in sublist]
    products_order_input = products[products['product_id'].isin(input)]
    products_order_input_name = products_order_input['product_name'].tolist()

    # Search for a similar basket in `b`
    basket = basket_recompose(order_w2v)
    return product_lift(basket, input), selection_w2v, products_order_input_name

# %%
## Interface ----
### Interface auxiliary functions ----

# Clean the terminal console
def clear_console(n=100):
    print('\n'*n)
    return None

# Request the user an input
def read_positive(message_try='Type a value', message_error='Only accepts values like `42`, try again.', color='input', allow_enter=False, max=100):
    # Loop until the user enters a value in the possibles
    correct_value = False
    while not correct_value:
        try:
            user_value = int(input(COLOR_CONSTANT[color] + message_try + COLOR_CONSTANT['end']))
            if (user_value > 0) & (user_value <= max):
                correct_value = True
            else:
                correct_value = False
                raise ValueError
        except ValueError:
            if allow_enter:
                # Allows to end the `while` when typed `enter` or other ilegal value
                correct_value = True
            else:
                correct_value = False
                print(COLOR_CONSTANT['error'] + message_error + COLOR_CONSTANT['end'])
    return user_value

def read_product_name(color='input'):
    # Loop until the user enters a value in the possibles
    message_try='Type a product name [ENTER to finish]: '
    input_text = input(COLOR_CONSTANT[color] + message_try + COLOR_CONSTANT['end'])
    if input_text != '':
        p = products[products['product_name'].str.contains(input_text.lower())].iloc[:5].reset_index()
    else:
        return False

    print_color('Choose from a product [Showing the first matching products]')
    print('\n')
    print(p['product_name'].reset_index().rename(columns={'index': 'Number', 'product_name': 'Product name'}).to_string(index=False))
    input_number = read_positive(message_try='Choose a number [ENTER to finish]: ', max = p.shape[0], allow_enter=True)
    if input_number:
        q = p.loc[input_number, ['product_name', 'product_id']]
        q = q['product_id']
        return q
    else:
        return False

def print_color(txt, color='note'):
    print(COLOR_CONSTANT[color] + txt + COLOR_CONSTANT['end'])

# %%
### Run user interface ----
clear_console()
tprint("Market basket analysis\nby Juan Pedro Bretti","Standard")

INPUT_TYPES = {1: 'Input product [Using name]', 2: 'Input product [Using ID]', 3: 'Input user', 4: 'Input users list', 5: 'Input order', 6: 'TSNE plots', 7: 'Auto EDA', 8: 'Orders and frequency EDA', 9: 'Exit'}
select_continue = 1  # Start the loop

# Infinite loop, will stop  by pressing Ctrl+C or selecting 'Exit' when prompt
while select_continue < max(INPUT_TYPES.keys()):
    # Prompt the user to continue or stop the infinite rounds
    select_continue = read_positive(message_try='Choose the type of input %s: ' % (
                                        ' '.join('\n{}: {}'.format(k, v) for k, v in INPUT_TYPES.items())),
                                    message_error='Only accepts the values %s, try again.' % (tuple(INPUT_TYPES.keys()),)
                                    )

    if select_continue < max(INPUT_TYPES.keys()) :
        clear_console()
        tprint(INPUT_TYPES[select_continue], "Standard")

    # Selection based on the key from `INPUT_TYPES`
    if select_continue == 1:
        # Multiple entries until the user press `Enter` to finish
        # https://www.geeksforgeeks.org/python-get-a-list-as-input-from-user/
        try:
            select_list = list()
            while True:
                select_ = read_product_name()
                if select_:
                    select_list.append(select_)
                else:
                    raise ValueError
        except:
            out_ = basket_input_product_list(select_list)
            print_color('Selected:')
            print(out_[1])
            print('\n')
            print_color('Recommendation:')
            print(out_[0])
            print('\n')
            print_color('Plot will open in Internet browser, please wait...')
            tsne_plot2(products, title='Selected `products` between others', selection=out_[2])

    if select_continue == 2:
        # Multiple entries until the user press `Enter` to finish
        # https://www.geeksforgeeks.org/python-get-a-list-as-input-from-user/
        try:
            select_list = list()
            while True:
                select_ = read_positive(message_try='Product number [ENTER to finish]: ', allow_enter=True)
                select_list.append(select_)
        except:
            out_ = basket_input_product_list(select_list)
            print_color('Selected:')
            print(out_[1])
            print('\n')
            print_color('Recommendation:')
            print(out_[0])
            print('\n')
            print_color('Plot will open in Internet browser, please wait...')
            tsne_plot2(products, title='Selected `products` between others', selection=out_[2])

    elif select_continue == 3:
        select_ = read_positive(message_try='Reference user: ')
        out_ = basket_input_user(select_)
        print_color('Similar to users:')
        print(out_[1])
        print('\n')
        print_color('Products purchased in previous orders:')
        print(out_[3])
        print('\n')
        print_color('Recommendation:')
        print(out_[0])
        print('\n')
        print_color('Plot will open in Internet browser, please wait...')
        tsne_plot2(df_user_basket, title='Selected `user` between others', selection=out_[2], hover='user_id')

    elif select_continue == 4:
        try:
            select_list = list()
            while True:
                select_ = read_positive(message_try='User number [e.g., [23, 27, 66]. ENTER to finish]: ', allow_enter=True)
                select_list.append(select_)
        except:
            out_ = basket_input_user_list(select_list)
            print_color('Selected:')
            print(out_[1])
            print('\n')
            print_color('Products purchased in previous orders:')
            print(out_[3])
            print('\n')
            print_color('Recommendation:')
            print(out_[0])
            print('\n')
            print_color('Plot will open in Internet browser, please wait...')
            tsne_plot2(df_user_basket, title='Selected `users` between others', selection=out_[2], hover='user_id')

    elif select_continue == 5:
        select_ = read_positive(message_try='Reference order: ')
        out_ = basket_input_order(select_)
        print_color('Products purchased in previous orders:')
        print(out_[2])
        print('\n')
        print_color('Recommendation:')
        print(out_[0])
        print('\n')
        print_color('Plot will open in Internet browser, please wait...')
        tsne_plot2(df_order_baskets, title='Selected `order` between others', selection=out_[1], hover='order_id')

    elif select_continue == 6:
        print_color('Please wait')
        # Create 3 different plots using TSNE algorithm
        tsne_plot(products, title='Products', color='department_id', product_flag=True)
        tsne_plot(df_user_basket, title='User average', color='user_id')
        tsne_plot(df_order_baskets, title='Order average', color='order_id')
        print_color('Done. Check your Internet browser.')

    elif select_continue == 7:
        print_color('Please wait')
        ProfileReport(products[['product_id', 'product_name', 'department', 'aisle']], title="Exploratory Data Analysis: `Products`").to_file("EDA/eda_products.html")
        ProfileReport(orders, title="Exploratory Data Analysis: `Orders and Users`").to_file("EDA/eda_orders_users.html")
        ProfileReport(orders_filter, title="Exploratory Data Analysis: `Orders and Products` (filtered)").to_file("EDA/eda_orders_products.html")
        print_color('Done. Check `EDA` folder.')

    elif select_continue == 8:
        print_color('Check popup window')
        eda(order_baskets, orders_filter, products)

# %%
