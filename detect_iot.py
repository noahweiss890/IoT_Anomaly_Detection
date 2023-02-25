# imports
import pandas as pd
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import xgboost as xgb
import gradio as gr


# Load data into a pandas dataframe
data = pd.read_csv('UNSW_NB15_training-set.csv')

# Remove all NAN columns or replace with desired string
# This loop iterates over all of the column names which are all NaN
for column in data.columns[data.isna().any()].tolist():
    [column] = data[column].fillna('None')

# Apply LabelEncoder to the entire dataframe
le = LabelEncoder()
data = data.apply(le.fit_transform)

# Remove some columns that may be needed for the model
data = data.drop(['is_sm_ips_ports', 'ct_srv_dst', 'ct_ftp_cmd', 
                     'ct_dst_src_ltm', 'ct_srv_src', 'dwin', 'swin', 
                     'sinpkt', 'dloss', 'sloss', 'sbytes', 'dbytes',
                     'spkts', 'dpkts'], axis=1)

# Load and preprocess your data
# X is your input data, y is your target labels
X = data.drop(['label', 'attack_cat'], axis=1).values
y = data['label']
# Normalize the datas
X = X / 255.

# Reshape the data if necessary
X = X.reshape((len(X), np.prod(X.shape[1:])))

X = X[:175341]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set the encoding dimensions
hidden_dim1 = 32
hidden_dim2 = 16
hidden_dim3 = 8

# Define the input layer
input_img = Input(shape=(29,))

# Define the encoder layers
hidden_layer_1 = Dense(hidden_dim1, activation='relu')(input_img)
hidden_layer_2 = Dense(hidden_dim2, activation='relu')(hidden_layer_1)
hidden_layer_3 = Dense(hidden_dim3, activation='relu')(hidden_layer_2)

# Define the decoder layers
decoder_hidden_1 = Dense(hidden_dim2, activation='relu')(hidden_layer_3)
decoder_hidden_2 = Dense(hidden_dim1, activation='relu')(hidden_layer_2)
decoder_output = Dense(29, activation='relu')(hidden_layer_1)

# Define the autoencoder model
autoencoder = Model(input_img, decoder_output)

# Get the encoder model
encoder = Model(input_img, hidden_layer_1)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
autoencoder.fit(X_train, X_train,
                epochs=50,
                batch_size=256,
                shuffle=True)

# Use the encoder to extract features from the data
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)

# Train a Random Forest classifier on the encoded features
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_model = xgb.XGBClassifier(random_state=42)
knn_model = KNeighborsClassifier()

# Create the voting classifier
voting_clf = VotingClassifier(estimators=[('xgb', xgb_model), ('rf', rf_model), ('knn', knn_model)], voting='hard')

print('Training the voting classifier...')
voting_clf.fit(X_train_encoded, y_train)
print('Done training the voting classifier...')

# Make predictions on the test data
y_pred = voting_clf.predict(X_test_encoded)

# Calculate the accuracy, recall, precision and f1-score of the classifier
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(' Accuracy: ', acc)
print('Recall: ', rec)
print('Precision: ', prec)
print('f1-score: ', f1)


def detect(id, dur, proto, service, state, spkts, dpkts, sbytes, dbytes, rate, sttl, dttl, sload, dload, sloss, dloss, sinpkt, dinpkt, sjit, djit, swin, stcpb, dtcpb, dwin, tcprtt, synack, ackdat, smean, dmean, trans_depth, response_body_len, ct_srv_src, ct_state_ttl, ct_dst_ltm, ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm, is_ftp_login, ct_ftp_cmd, ct_flw_http_mthd, ct_src_ltm, ct_srv_dst, is_sm_ips_ports):
    
    x = np.array([id, dur, proto, service, state, rate, sttl, dttl, sload, dload, dinpkt, sjit, djit, stcpb, dtcpb, tcprtt, synack, ackdat, smean, dmean, trans_depth, response_body_len, ct_state_ttl, ct_dst_ltm, ct_src_dport_ltm, ct_dst_sport_ltm, is_ftp_login, ct_flw_http_mthd, ct_src_ltm])
    
    x = le.fit_transform(x)
    x = x / 255.
    x = encoder.predict(x.reshape(1, -1))
    
    prediction = voting_clf.predict(x.reshape(1, -1))
    if prediction[0] == 0:
        return "Normal"
    else:
        return "Anomaly"

outputs = gr.outputs.Textbox()
inputs = ['number']*43
inputs[2] = 'text'
inputs[3] = 'text'
inputs[4] = 'text'
app = gr.Interface(fn=detect, inputs=inputs, outputs=outputs,description="This is a IoT Anommaly Detection Model")

app.launch(share=True, server_name="0.0.0.0", server_port=8080)