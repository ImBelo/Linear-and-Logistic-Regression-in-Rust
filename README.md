Linear and Logistic Regression in Rust!!
Why Rust? Because I need to learn it, and there's no better way than building something real!

Implemented a simple linear regression model (predicting 2x + 1) and a logistic regression model that predicts gender from height and weight. (Cancel me if you want, it's just math! 😄)

📚 WHAT I LEARNED
1. Feature Scaling is NOT Optional
The learning rate is not scale invariant! I discovered this the hard way when my weights exploded to -861.25 while accuracy stayed stuck at 54.70%. Two solutions:

Normalize the data (what I chose) - Center features to mean=0, std=1

Per-feature learning rates - Too complex, don't do this

Before normalization:

text
Weights: height=-861.2522, weight=279.4597, bias=-8.5124  😱
Accuracy: 54.70% (stuck!)
After normalization:

text
Weights: height=0.5727, weight=0.5371, bias=-0.5207  😊
Accuracy: 81.94% and improving!
2. Polars + Arc = Thread Safety
Polars uses multi-threading under the hood for blazing fast performance. That's why you need to wrap the Schema in Arc:

3. The Bias is Special
When normalizing features, never normalize the bias term! The bias represents the baseline prediction when all features are at their mean:

rust
// After normalization (mean=0, std=1):
// features = [0.0, 0.0] represents an "average" person

// The bias now directly tells you:
// P(Female) = sigmoid(bias) for an average person

// If you normalized the bias, this interpretation breaks!
