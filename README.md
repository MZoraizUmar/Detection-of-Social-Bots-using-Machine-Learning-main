🤖 AI-Based Detection of Malicious Bot Activity on Twitter
Welcome to AI Tweet Detector, a cutting-edge system that identifies whether a tweet is posted by a human or a bot using advanced AI/ML models like BERT, LSTM, Random Forest, XGBoost, and more!

📝 Project Overview
In today's digital world, malicious bots manipulate public opinion and trends across social media platforms. This project develops a hybrid AI system that accurately detects bot-generated content on Twitter, enhancing security and trust.

Before Running Download the Dataset cresci-2017 from https://botometer.osome.iu.edu/bot-repository/datasets.html

✅ Models Used:

BERT (Transformer-Based Model)

LSTM (Long Short-Term Memory Neural Networks)

Random Forest Classifier

XGBoost Classifier

Naive Bayes Classifier

Logistic Regression

✅ Real-Time Statistics:

Total Tweets Analyzed

Number of Humans Detected

Number of Bots Detected

Dynamic Pie Charts, Bar Graphs, and Accuracy Cards

✅ Beautiful Front-End:

Built with Flask + Bootstrap 5

Professional dashboard, detector page, animated graphs

Model-wise comparison of predictions

Final hybrid decision (majority voting + confidence scores)

# 📂 Project Structure — Detection of Social Bots using Machine Learning

Detection-of-Social-Bots-using-Machine-Learning-main/
│
├── app.py # Main Flask backend server for routing and API logic
│
├── templates/ # HTML templates for the frontend interface
│ ├── index.html # Home page
│ ├── about.html # About the project and team
│ ├── services.html # Services or feature overview
│ ├── dashboard.html # Analytics dashboard for visual stats
│ └── detector.html # Tweet input form with prediction result display
│
├── static/ # Static assets (CSS, JavaScript, images)
│ ├── styles.css # Stylesheet for UI design
│ ├── images/ # Folder containing logo, icons, etc.
│ └── animations/ # Folder for any Lottie or GIF animations
│
├── processed/ # Pretrained models, tokenizers, and vectorizers
│ ├── bert_model.pt # PyTorch BERT model for text classification
│ ├── lstm_model.h5 # Keras-based LSTM model
│ ├── RandomForest_model.pkl # Pickled Random Forest model
│ ├── XGBoost_model.pkl # Pickled XGBoost model
│ ├── LogisticRegression_model.pkl# Pickled Logistic Regression model
│ ├── NaiveBayes_model.pkl # Pickled Naive Bayes model
│ ├── tokenizer.pkl # Tokenizer for deep learning models
│ └── tfidf_vectorizer.pkl # TF-IDF vectorizer for classical ML models
│
└── README.md # Project documentation, instructions, and overview

🚀 How to Run Locally
Clone the repo:


git clone https://github.com/MZoraizUmar/Detection-of-Social-Bots-using-Machine-Learning-main.git
cd Detection-of-Social-Bots-using-Machine-Learning-main
Install required libraries:


pip install -r requirements.txt
Run the Flask app:


python app.py
Open your browser and go to:


http://127.0.0.1:5000/
⚙️ Requirements
Python 3.10+

Flask

TensorFlow

PyTorch

Transformers (Huggingface)

Scikit-learn

XGBoost

Joblib

Matplotlib / Chart.js (for front-end graphs)

Install all using:


pip install -r requirements.txt
📈 Results
✅ Achieved 95%+ accuracy on detecting bots
✅ Hybrid Ensemble Model boosts decision reliability
✅ Real-time dynamic statistics tracking
✅ Interactive, modern UI/UX

👨‍💻 Team Members
Muhammad Muhibullah

Ali Haider

Pooja Dhaduk

Nushrat Jahan

Muhammad Zoraiz Umar
(University of Wollongong)

📚 License
This project is licensed under the MIT License - see the LICENSE file for details.

🌟 Acknowledgements
Thanks to University of Wollongong

Special thanks to Supervisor: Partha Sarathi

Huggingface Transformers

TensorFlow & PyTorch community

🏁 Future Work
Deploy on AWS/Heroku

Extend to multi-language Tweet detection

Build a Chrome Extension for real-time bot detection

Integrate live Twitter feed monitoring 📡

🚀 Let's Make Twitter Safer Together! 🤖😎
