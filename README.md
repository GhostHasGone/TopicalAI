# Topical AI - Automatic Rule-Breaking Message Detection

## Overview
Topical AI is a machine learning-based project designed to detect and classify rule-breaking messages in text data. The primary use case is to identify inappropriate or unwanted content in online communities like Discord. This project uses a `RandomForestClassifier` model trained on labeled data to classify messages into predefined categories.

## Features
- **Message Classification**: Classifies messages into categories such as `acceptable`, `buying_account`, and `asking_for_boost`.
- **Class Imbalance Handling**: Uses SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset.
- **Performance Metrics**: Outputs accuracy, precision, recall, and F1-score for the trained model.
- **Batch Message Testing**: Reads messages from a file and classifies them, providing detailed statistics on rule-breaking messages.
- **Customizable Dataset**: Easily modify the dataset to include new categories or adjust existing ones.

## Requirements
- Python 3.8 or higher
- Required Python libraries:
  - `pandas`
  - `scikit-learn`
  - `imbalanced-learn`
  - `datetime`

Install the required libraries using:
```bash
pip install pandas scikit-learn imbalanced-learn
```

## File Structure
- `main.py`: The main script for training the model and classifying messages.
- `messages.csv`: The dataset containing labeled messages for training and testing.
- `test_messages.txt`: A file containing messages to be classified by the trained model.
- `README.md`: Documentation for the project.

## Dataset
The dataset (`messages.csv`) contains two columns:
- `message`: The text of the message.
- `label`: The category of the message (`acceptable`, `buying_account`, `asking_for_boost`).

Example:
```csv
message,label
Looking to buy an account,buying_account
Can someone boost my stats for me?,asking_for_boost
Let's play together this weekend,acceptable
```

## How It Works
1. **Data Loading**: The dataset is loaded from `messages.csv`.
2. **Text Vectorization**: Messages are transformed into numerical representations using `TfidfVectorizer`.
3. **Class Imbalance Handling**: SMOTE is applied to balance the dataset by oversampling the minority classes.
4. **Model Training**: A `RandomForestClassifier` is trained on the processed data.
5. **Performance Evaluation**: The model's performance is evaluated using accuracy, precision, recall, and F1-score.
6. **Message Classification**: The trained model is used to classify messages from `test_messages.txt`.

## Usage

### Training the Model
Run the `main.py` script to train the model and evaluate its performance:
```bash
python main.py
```

### Testing Messages
To test new messages:
1. Add each message you want to classify to the `test_messages.txt` file (one per line).
2. Run the script, and it will classify each message and display results in the console.

If a message is classified incorrectly:
- Open `messages.csv`.
- Add the misclassified message as a new row with the correct label.
- Re-run the training script (`python main.py`) to update the model with the improved data.

This continuous feedback loop helps improve accuracy over time.

### Output
The script provides:
- Total messages checked.
- Count and percentage of messages flagged as `buying_account` or `asking_for_boost`.
- Total rule-breaking messages and their percentage.
- Time taken for classification.

Example Output:
```
Total Messages Checked: 100
Buying / Selling account: 10 (10.00%)
Asking for boost: 5 (5.00%)
Total Rule Breaking Messages: 15 / 100 (15.00%)
Total Time Taken: 0:00:01.2345
Time Taken Per Message: 0.0123 seconds
```

## Customization
- **Adding New Categories**: Update the `messages.csv` file with new labels and retrain the model.
- **Adjusting SMOTE Parameters**: Modify the `sampling_strategy` and `k_neighbors` parameters in the SMOTE initialization to fine-tune oversampling.
- **Changing the Model**: Replace the `RandomForestClassifier` with another classifier from `scikit-learn` if needed.

## Notes
- Ensure the dataset is properly labeled for accurate classification.
- The bot's performance depends on the quality and diversity of the training data.
- For deployment in real-time applications (e.g., Discord bots), integrate the `classify_message` function into your bot's message handling logic.

## License
This project is for educational purposes and is not intended for production use. Modify and use it at your own discretion.

## Acknowledgments
- **scikit-learn**: For providing tools for machine learning and data preprocessing.
- **imbalanced-learn**: For the SMOTE implementation to handle class imbalance.
