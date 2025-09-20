# Online Learning with River 🚀

Explore the power of online machine learning with River - a Python library for real-time ML. This project demonstrates how online learning models can adapt to concept drift in streaming data.

## Why This Matters

🔄 **Continuous Learning**: Models update in real-time as new data arrives  
🎯 **Adaptive**: Automatically adjusts to changing data distributions  
⚡ **Efficient**: Processes data one instance at a time, no retraining needed  
📊 **Visualized**: Interactive Streamlit dashboard to track model performance  
🔍 **Transparent**: See model predictions and confidence scores in action  

## Key Features

- **Real-time Model Adaptation**: Watch as the model learns from streaming data
- **Concept Drift Simulation**: Demonstrates handling of changing data patterns
- **Confidence Scoring**: See prediction confidence levels for each decision
- **Performance Comparison**: Compare online vs batch learning approaches
- **Interactive Dashboard**: Control and visualize the learning process

## Installation and Setup

### Prerequisites
- Python 3.8+
- uv (recommended) or pip

### 1. Install Dependencies

Using uv (recommended):
```bash
uv sync
```

Using pip:
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit App
```bash
streamlit run main.py
```

## Project Structure

```bash
online-machine-learning-with-river/
├── main.py              # Main Streamlit application
├── requirements.txt     # Project dependencies
├── pyproject.toml      # Project configuration
├── uv.lock            # uv lock file
└── README.md           # This file
```

## Usage
- Start the application using the command above
- Configure the simulation parameters
- Run the simulation to see online learning in action
- Analyze the model's performance and adaptation
- Experiment with different data distributions and model settings

## How It Works
The application demonstrates online learning using River's Naive Bayes classifier. It shows:

- How models can learn from streaming data
- The impact of concept drift on model performance
- The advantage of online learning in dynamic environments

## Contribution
Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## License
This project is open source and available under the MIT License.