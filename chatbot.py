import re
import openai
import os

# Set up your OpenAI API key (replace "YOUR_API_KEY_HERE" with your actual key directly or use environment variables)
openai.api_key = "YOUR_API_KEY_HERE"

# Define rule-based patterns and responses
rules = [
    {"pattern": r"\bhello\b|\bhi\b|\bhey\b", "response": "Hello! How can I assist you today?"},
    {"pattern": r"\bhow are you\b", "response": "I'm here to help! How can I assist you?"},
    {"pattern": r"\bwhat is your name\b", "response": "I am a chatbot created to help you. You can call me ChatBot!"},
    {"pattern": r"\bbye\b|\bgoodbye\b", "response": "Goodbye! Feel free to reach out again if you need help."},
    {"pattern": r"\bhello\b|\bhi\b|\bhey\b", "response": "Hello! How can I assist you today?"},
    {"pattern": r"\bhow are you\b", "response": "I'm here to help! How can I assist you?"},
    {"pattern": r"\bwhat is your name\b", "response": "I am a chatbot created to help you. You can call me ChatBot!"},
    {"pattern": r"\bbye\b|\bgoodbye\b", "response": "Goodbye! Feel free to reach out again if you need help."},
    {"pattern": r"\bwhat can you do\b", "response": "I can answer questions, provide information, and assist with various tasks. How can I help you today?"},
    {"pattern": r"\bthank you\b|\bthanks\b", "response": "You're welcome! I'm happy to help."},
    {"pattern": r"\bhow old are you\b", "response": "I don't have an age. I'm a chatbot created to assist you!"},
    {"pattern": r"\bwhat is the time\b", "response": "I can't provide real-time info, but you can check your device for the current time."},
    {"pattern": r"\bwhat is the weather like\b", "response": "Sorry, I cannot check the weather. You can use a weather app or website for the latest updates."},
    {"pattern": r"\bcan you help me\b", "response": "Of course! What do you need help with?"},
    {"pattern": r"\bhow do I use this chatbot\b", "response": "You can ask me questions or request help with various tasks. Just type your request, and I'll do my best to assist you!"},
    {"pattern": r"\bi need help\b", "response": "I'm here to help! What can I assist you with?"},
    {"pattern": r"\bwho created you\b", "response": "I was created by a team of developers to assist you with various tasks."},
    {"pattern": r"\bcan you tell a joke\b", "response": "Sure! Why don't skeletons fight each other? They don't have the guts!"},
    {"pattern": r"\bwhat do you know\b", "response": "I can provide information on a variety of topics. Just ask away!"},
    {"pattern": r"\bwhat is 2 plus 2\b", "response": "2 plus 2 is 4."},
    {"pattern": r"\bhow can I contact support\b", "response": "You can contact support by visiting our website or emailing support@example.com."},
    {"pattern": r"\bwhere are you from\b", "response": "I don't have a physical location, but I'm here to help wherever you are."},
    {"pattern": r"\bdo you have emotions\b", "response": "I don't have emotions, but I can understand your queries and try to help as best as I can."},
    {"pattern": r"\bcan you search the internet\b", "response": "I cannot browse the web, but I can provide information based on the knowledge I've been trained on."},
    {"pattern": r"\bwhat is AI\b", "response": "AI, or Artificial Intelligence, is a field of computer science that focuses on creating systems capable of performing tasks that normally require human intelligence."},
    {"pattern": r"\bwho is the president of the USA\b", "response": "I cannot provide real-time information. Please check the latest news for updates."},
    {"pattern": r"\bwhat is your purpose\b", "response": "My purpose is to assist you with your questions and tasks, making things easier for you."},
    {"pattern": r"\bcan you do calculations\b", "response": "Yes, I can help with simple math! Just ask away."},
    {"pattern": r"\bare you human\b", "response": "No, I'm a chatbot, not human. But I'm here to help with anything you need!"},
    {"pattern": r"\bwhat is a chatbot\b", "response": "A chatbot is a software application designed to simulate conversation with users, especially over the internet."},
    {"pattern": r"\bwhat is your favorite color\b", "response": "I don't have personal preferences, but blue is a popular color!"},
    {"pattern": r"\bhow do I install a program\b", "response": "To install a program, visit the software's website, download the installer, and follow the on-screen instructions."},
    {"pattern": r"\bwhat is programming\b", "response": "Programming is the process of writing instructions that a computer can execute to perform specific tasks."},
    {"pattern": r"\bwhat is Python\b", "response": "Python is a popular, high-level programming language known for its simplicity and readability."},
    {"pattern": r"\bcan you translate languages\b", "response": "I can translate short phrases into different languages. Just let me know what you need translated."},
    {"pattern": r"\bwhat is 5 times 5\b", "response": "5 times 5 is 25."},
    {"pattern": r"\bdo you have a favorite movie\b", "response": "I don't watch movies, but many people love classics like The Shawshank Redemption or Inception!"},
    {"pattern": r"\bcan you play music\b", "response": "I cannot play music, but I can suggest playlists or songs if you'd like."},
    {"pattern": r"\bhow do I reset my password\b", "response": "To reset your password, go to the login page and look for the 'Forgot Password' link to receive a reset email."},
    {"pattern": r"\bwhat is your favorite food\b", "response": "I don't eat, but pizza is a favorite for many people!"},
    {"pattern": r"\bcan you provide news updates\b", "response": "I cannot provide real-time news updates, but I can suggest websites to check for the latest news."},
    {"pattern": r"\bwhat is a computer\b", "response": "A computer is an electronic device that can store, retrieve, and process data, allowing users to perform various tasks."},
    {"pattern": r"\bdo you understand emotions\b", "response": "I can recognize emotional expressions in text, but I don't experience emotions myself."},
    {"pattern": r"\bwhat is machine learning\b", "response": "Machine learning is a subset of artificial intelligence that allows computers to learn from data and improve over time without explicit programming."},
    {"pattern": r"\bwho is the CEO of Tesla\b", "response": "As of the latest information, Elon Musk is the CEO of Tesla."},
    {"pattern": r"\bwhat is 10 divided by 2\b", "response": "10 divided by 2 is 5."},
    {"pattern": r"\bcan you set a reminder\b", "response": "I can't set reminders, but you can use your phone or a calendar app to do that."},
    {"pattern": r"\bdo you play games\b", "response": "I can't play games, but I can help you find some fun games to try!"},
    {"pattern": r"\bhow can I learn to code\b", "response": "There are many resources online to learn coding, including websites like Codecademy, FreeCodeCamp, and Coursera."},
    {"pattern": r"\bcan you solve riddles\b", "response": "Sure! Here's a riddle: What has keys but can't open locks? (Answer: A piano!)"},
    {"pattern": r"\bwhat is 100 minus 25\b", "response": "100 minus 25 is 75."},
    {"pattern": r"\bwhat is the capital of France\b", "response": "The capital of France is Paris."},
    {"pattern": r"\bcan you help with homework\b", "response": "I can help you understand concepts and answer questions, but I won't do the homework for you!"},
    {"pattern": r"\bhello\b|\bhi\b|\bhey\b", "response": "Hello! How can I assist you today?"},
    {"pattern": r"\bhow are you\b", "response": "I'm here to help! How can I assist you?"},
    {"pattern": r"\bwhat is your name\b", "response": "I am a chatbot created to help you. You can call me ChatBot!"},
    {"pattern": r"\bbye\b|\bgoodbye\b", "response": "Goodbye! Feel free to reach out again if you need help."},
    {"pattern": r"\bwhat can you do\b", "response": "I can answer questions, provide information, and assist with various tasks. How can I help you today?"},
    {"pattern": r"\bthank you\b|\bthanks\b", "response": "You're welcome! I'm happy to help."},
    {"pattern": r"\bhow do I get started with AI\b", "response": "To get started with AI, you can begin by learning Python and exploring machine learning frameworks like TensorFlow or PyTorch."},
    {"pattern": r"\bwhat is machine learning\b", "response": "Machine learning is a subset of artificial intelligence that enables computers to learn from data and improve over time without explicit programming."},
    {"pattern": r"\bwhat is deep learning\b", "response": "Deep learning is a subset of machine learning that uses neural networks with many layers (hence 'deep') to analyze various forms of data."},
    {"pattern": r"\bwhat is a neural network\b", "response": "A neural network is a series of algorithms that attempt to recognize underlying relationships in a set of data by mimicking how the human brain operates."},
    {"pattern": r"\bwhat is supervised learning\b", "response": "Supervised learning is a type of machine learning where the model is trained on labeled data. It learns from input-output pairs to make predictions."},
    {"pattern": r"\bwhat is unsupervised learning\b", "response": "Unsupervised learning involves training a model on data without labels. The goal is to find hidden patterns or groupings in the data."},
    {"pattern": r"\bwhat is reinforcement learning\b", "response": "Reinforcement learning is a type of machine learning where an agent learns by interacting with its environment and receiving feedback in the form of rewards or penalties."},
    {"pattern": r"\bwhat is a convolutional neural network\b", "response": "A Convolutional Neural Network (CNN) is a type of deep neural network used primarily for image recognition and processing."},
    {"pattern": r"\bwhat is natural language processing\b", "response": "Natural Language Processing (NLP) is a branch of AI that focuses on the interaction between computers and human (natural) languages."},
    {"pattern": r"\bwhat is transfer learning\b", "response": "Transfer learning involves taking a model that has been trained on one task and fine-tuning it for a different but related task."},
    {"pattern": r"\bcan you explain overfitting in machine learning\b", "response": "Overfitting occurs when a model learns the details and noise in the training data to the extent that it negatively impacts its performance on new data."},
    {"pattern": r"\bwhat is feature engineering\b", "response": "Feature engineering is the process of using domain knowledge to select, modify, or create new features from raw data to improve model performance."},
    {"pattern": r"\bwhat is the purpose of regularization in machine learning\b", "response": "Regularization is a technique used to prevent overfitting by adding a penalty to the model's complexity, ensuring it generalizes well to new data."},
    {"pattern": r"\bwhat is gradient descent\b", "response": "Gradient descent is an optimization algorithm used to minimize the loss function by iteratively adjusting the parameters of a model."},
    {"pattern": r"\bcan you explain backpropagation\b", "response": "Backpropagation is the process used in neural networks to update the weights of the network by calculating the gradient of the loss function with respect to each weight."},
    {"pattern": r"\bwhat is the difference between AI and machine learning\b", "response": "AI is the broader field that encompasses any technique that enables computers to mimic human intelligence, while machine learning is a subset of AI that uses data to allow machines to learn and improve automatically."},
    {"pattern": r"\bwhat is a support vector machine\b", "response": "A Support Vector Machine (SVM) is a supervised learning algorithm used for classification and regression tasks by finding the hyperplane that best separates the data."},
    {"pattern": r"\bwhat is a decision tree\b", "response": "A decision tree is a supervised learning algorithm used for classification and regression tasks. It splits data into branches based on feature values to make predictions."},
    {"pattern": r"\bcan you help me with my AI project\b", "response": "Of course! What kind of AI project are you working on? I can assist with suggestions, ideas, or solving specific challenges."},
    {"pattern": r"\bhow do I build an AI model\b", "response": "To build an AI model, start by collecting and preparing data, choosing the right algorithm, training the model, and evaluating its performance. I can help guide you through these steps."},
    {"pattern": r"\bwhat is a training dataset\b", "response": "A training dataset is a set of data used to train a machine learning model. The model learns patterns from this data to make predictions."},
    {"pattern": r"\bwhat is cross-validation in machine learning\b", "response": "Cross-validation is a technique used to evaluate the performance of a machine learning model by dividing the dataset into multiple subsets and training/testing the model on different combinations."},
    {"pattern": r"\bwhat are hyperparameters in machine learning\b", "response": "Hyperparameters are parameters that are set before training a model, such as the learning rate, batch size, and the number of layers in a neural network."},
    {"pattern": r"\bhow do I deploy a machine learning model\b", "response": "To deploy a machine learning model, you can use cloud platforms like AWS, Google Cloud, or Azure to host and serve your model via an API or integrate it into a software application."},
    {"pattern": r"\bwhat is a random forest model\b", "response": "A random forest is an ensemble learning method that builds multiple decision trees and combines their outputs to make more accurate predictions."},
    {"pattern": r"\bwhat is PCA (Principal Component Analysis)\b", "response": "Principal Component Analysis (PCA) is a dimensionality reduction technique used to reduce the number of features in a dataset while retaining as much variability as possible."},
    {"pattern": r"\bwhat is a generative adversarial network\b", "response": "A Generative Adversarial Network (GAN) is a deep learning architecture that consists of two neural networks, a generator and a discriminator, that compete to generate new data similar to a training dataset."},
    {"pattern": r"\bhow do I choose the right algorithm for my project\b", "response": "Choosing the right algorithm depends on the nature of your data, the problem you're solving, and the desired outcomes. I can help you select based on your project's specifics."},
    {"pattern": r"\bcan you explain clustering algorithms\b", "response": "Clustering algorithms group similar data points together. Popular clustering algorithms include K-means, hierarchical clustering, and DBSCAN."},
    {"pattern": r"\bwhat is the difference between classification and regression\b", "response": "Classification is used for predicting categorical labels, while regression is used for predicting continuous values."},
    {"pattern": r"\bwhat is a chatbot project\b", "response": "A chatbot project involves creating an AI system that can simulate conversation with users. It typically involves NLP techniques to process and understand user input."},
    {"pattern": r"\bcan you help with a deep learning project\b", "response": "Absolutely! What kind of deep learning project are you working on? I can help with advice, frameworks, or code examples."},
    {"pattern": r"\bhow do I evaluate the performance of my machine learning model\b", "response": "You can evaluate your model's performance using metrics like accuracy, precision, recall, F1 score, and the confusion matrix, depending on your task (classification or regression)."},
    {"pattern": r"\bcan you help me with AI project documentation\b", "response": "I can help you with AI project documentation, including setting up clear goals, data descriptions, model architecture, and evaluation metrics."},
    {"pattern": r"\bwhat is the role of data preprocessing in machine learning\b", "response": "Data preprocessing involves cleaning and transforming raw data into a suitable format for machine learning models. It includes handling missing values, encoding categorical variables, and scaling numerical features."},
    {"pattern": r"\bwhat is the difference between bagging and boosting\b", "response": "Bagging (Bootstrap Aggregating) involves training multiple models independently and combining their outputs, while boosting involves training models sequentially, each trying to correct the errors of the previous one."},
    {"pattern": r"\bcan you explain model interpretability\b", "response": "Model interpretability refers to the ability to understand how a machine learning model makes predictions. It helps in trust, debugging, and ensuring fairness."},
    {"pattern": r"\bcan you suggest AI research papers\b", "response": "Sure! You can explore papers on platforms like arXiv, Google Scholar, and ResearchGate. I can suggest topics based on your area of interest."},
]


def rule_based_response(user_input):
    """Check user input against predefined patterns for a rule-based response."""
    for rule in rules:
        if re.search(rule["pattern"], user_input, re.IGNORECASE):
            return rule["response"]
    return None  # No rule matched

def get_chatgpt_response(user_input):
    """Use OpenAI's ChatGPT API to get a response for unmatched queries."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input}
        ],
        max_tokens=50,
        temperature=0.7
    )
    return response.choices[0].message['content'].strip()

# Chat loop
print("ChatBot: Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("ChatBot: Goodbye!")
        break

    # Get rule-based response
    response = rule_based_response(user_input)

    # Fallback to ChatGPT API if no rule matches
    if response is None:
        response = get_chatgpt_response(user_input)

    print("ChatBot:", response)
