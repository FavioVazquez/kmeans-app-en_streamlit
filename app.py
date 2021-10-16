import kmeans
import templates

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

import streamlit as st	
import streamlit.components.v1 as components

import plotly.graph_objects as go

st.set_page_config(page_title="K-means Visualization App", page_icon="python.png", layout="wide")

st.image("https://www.isb.edu/content/dam/sites/isb/about-isb/ISB_transparent_logo.png", width=400)

st.header('**Visualizing K-means step-by-step with Python**')
st.write("")

# components.html('<b>texto</b>')
# st.markdown("<h1 style='text-align: center; color: red;'>texto</h1>", unsafe_allow_html=True)

st.sidebar.title('Parameters')

with st.sidebar.beta_container():
   _, slider_col, _ = st.beta_columns([0.02, 0.96, 0.02])
   with slider_col:
        k = st.sidebar.select_slider(
				label='Number of simulated clusters (groups):',
				options=range(2,11), value=2
			)
        std = st.sidebar.slider(
				'Standard Deviation of simulated data:',
				0.1, 5.0, 1.0, 0.1
			)

mode = st.sidebar.selectbox('Method for initialization of the centroids:', ["random", "kmeans++"])

_, central_button, _ = st.sidebar.beta_columns([0.25, 0.5, 0.25])
with central_button:
	st.text("")
	st.button('Recompute')

st.markdown("""
For better comprehension of the technique, please read the explanation below and visualize its animated steps.

Try to modify the sample size, the number of clusters (K), and the standard deviation.
Notice that the fewer the number of groups and standard deviation, the faster we reach the final positions of the centroids.
The `kmeans++` algorithm also tends to accelerate this process.

""")

with st.beta_expander(label="Description and explanation"):
	
	st.markdown("""
K-means is an unsupervised machine learning algorithm aiming to identify clusters (i.e., groups) in the dataset. Like any other unsupervised model, its objective is to **identify patterns in the data** to interpret them. Therefore, **it's not suitable for prediction analysis**. It's commonly used for customer segmentation.

This model takes **only numerical variables** to cluster the observations in "K" clusters, where K means the number of clusters. K should be pre-defined before running the model. There are techniques that help us to identify the best number of groups in our data, like the *elbow* and the *silhouette* methods.

Avoid converting numerical to categorical variables if you don't have a reasonable theoretical basis to do so. For categorical variables, we can use K-modes, a variation of K-means also aiming to cluster our data, but using categorical variables instead. Since it's not the objective of this showcase, I'm leaving [this link](https://www.analyticsvidhya.com/blog/2021/06/kmodes-clustering-algorithm-for-categorical-data/) with a further explanation about this model.

In K-means, the categorical variables can be used to describe the identified clusters so that we can identify patterns in each cluster. It usually helps to understand our data and make better decisions.

K-means calculates the distance - usually, the euclidean distance - between the data points and the **centroids**, where the centroids are K points randomly distributed in the data, representing the center of each cluster. Therefore, since it's a distance-based technique, we need to standardize our data if there're variables with distinct scales. 

Here are the general steps of K-means:

1. Define the number of clusters (k);
2. Initialize the k centroids;
3. Label each data point to its nearest centroid;
4. Move (or "update") the centroid position to the mean of the points labeled to it;
5. Repeat steps 3 and 4 until the mean of each centroid doesn't change anymore or the maximum allowed interaction is reached.

&nbsp;

The centroid initialization can be at random or optimized with an algorithm called K-means++. By default, `scikit-learn` uses k-means++.

At random initialization, we randomly choose K data points to use as our centroids, or we can place random centroids within the dimension of our data points. On the other hand, k-means++ aims to initialize the centroids as far as possible from each other. It's usually good because it avoids the bias of random initialization of the centroids (see K-means trap). Furthermore, kmeans++ usually reduces the number of needed interactions to reach the final centroid positions.

Additionally, some applications run the K-means algorithm multiple times with different initialization start points, providing as output the one with the lesser variance within the groups. 
	
_**Notes:**_

---

Data used in this application has been simulated with [`datasets.make_blobs()`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html) from [`scikit-learn`](https://scikit-learn.org/stable/index.html). Here, I've restricted the number of both observations (100) and variables (2) so that the application doesn't get slow and we can visualize the graphic bidimensionally.

Furthermore, in the elbow method, only one run of K-means is executed per K (see [elbow method](https://www.oreilly.com/library/view/statistics-for-machine/9781788295758/c71ea970-0f3c-4973-8d3a-b09a7a6553c1.xhtml)), which can result in non-standard less of distortion, since some K's could be a result of a poor centroid initialization.

Suggestions of critics? Message me via [LinkedIn](https://www.linkedin.com/in/cmcouto-silva/).

_**Code availability:**_

---

The scripts with the step-by-step implementation of K-means and this are available at GitHub:

- [App repository](https://github.com/cmcouto-silva/kmeans-app-en_streamlit)
- [Step-by-step K-means' script (no scikit-learn)](https://github.com/cmcouto-silva/kmeans-app-en_streamlit/blob/main/kmeans.py)

&nbsp;
""")

data = make_blobs(centers=k, cluster_std=std)
df = pd.DataFrame(data[0], columns=['x','y']).assign(label = data[1])

if st.checkbox('Show raw data'):
	_, df_col, _ = st.beta_columns([0.25,0.2, 0.25])
	with df_col:
		st.write(df)

model, wss = kmeans.calculate_WSS(data[0], k, 10, mode=mode)
raw_col, elbow_col = st.beta_columns([0.5,0.5])

_, kanimation_col, _ = st.beta_columns([0.2,0.8,0.2])

with kanimation_col:
	fig = kmeans.plot(model)
	fig = fig.update_layout(autosize=False, height=560,
		title_text="<b>Visualizing K-means - animated steps</b>", title_font=dict(size=24))
	st.plotly_chart(fig, use_container_width=True, sharing="streamlit")

with raw_col:
	raw_fig = go.Figure(
		data=fig.data[0],
		layout=dict(
			template='seaborn', title='<b>Unlabeled data</b>',
			xaxis=dict({'title':'x'}), yaxis=dict({'title':'y'})
			)
		)
	st.plotly_chart(raw_fig, use_container_width=True)

with elbow_col:
	elbow_fig = go.Figure(
	data=go.Scatter(x=list(range(1,11)), y=wss),
	layout=dict(
		template='seaborn', title='<b>Elbow method</b>',
		xaxis=dict({'title':'k'}), yaxis=dict({'title':'wss'})
		)
	)
	st.plotly_chart(elbow_fig, use_container_width=True)


st.markdown("""
---

## **K-means trap**

The random initialization of the centroids might result in non-representative groups. In the example below, there are four groups (to the left), 
but when applied K-means with the random initialization (on right), the centroids started near to each other and resulted in non-representative groups 
when compared to the observed data. That's why we usually apply K-means with `kmeans++` algorithm, 
also repeating the full K-means procedure a couple of times to get the one with less variability per group. 

Finally, it's worth noticing that there are distinct and more complex grouping structures where K-means is not the best model to identify the groups.
In such cases, we use other clustering models.

The first figure of the [scikit-learn clustering section](https://scikit-learn.org/stable/modules/clustering.html) shows an example of
how each of the implemented clustering models performs on different data structures.

&nbsp;

""")

# Specific biased data
raw_seed, kanimation_seed = st.beta_columns([0.5,0.5])

data_seed,labels_seed = make_blobs(centers=4, random_state=3)
model_seed = kmeans.Kmeans(data_seed, 4, seed=2)
model_seed.fit()

with raw_seed:
	raw_fig_seed = go.Figure(
		data=go.Scatter(x=data_seed[:,0], y=data_seed[:,1], mode='markers', marker=dict(color=labels_seed)),
		layout=dict(title_text="<b>Labeled points according to the actual clusters</b>",
			template="simple_white", title_font=dict(size=18)))
	raw_fig_seed.update_layout(templates.simple_white, height=500, title_x=0.15, title_font_size=18)
	st.plotly_chart(raw_fig_seed, use_container_width=True, sharing="streamlit")

with kanimation_seed:
	fig_seed = kmeans.plot(model_seed)
	fig_seed = fig_seed.update_layout(autosize=False, height=500,
		title_text="<b>Visualizing bias in centroid initialization</b>", title_font=dict(size=21))
	st.plotly_chart(fig_seed, use_container_width=True, sharing="streamlit")


# st.markdown("""For suggestions of critics, please message me through [LinkedIn](https://www.linkedin.com/in/cmcouto-silva/).""")
