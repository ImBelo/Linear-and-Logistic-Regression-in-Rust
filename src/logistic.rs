use ndarray::{ArrayView1, ArrayView2, Array1,Array2,s};
use polars::prelude::*;
use crate::process_data::preprocess_data;

struct Logistic {
    weights: Array1<f64>,
    // we treat bias separatly because we dont want to normalize it and be penilized
    bias: f64,
    learning_rate: f64,
}

impl Logistic {
    fn new(n_features: usize, learning_rate: f64) -> Self {
        Self {
            weights: Array1::zeros(n_features),
            bias: 0.0,
            learning_rate,
        }
    }
    
    fn train(&mut self, features: &ArrayView2<f64>, targets: &ArrayView1<f64>) {
        let n_samples = features.nrows() as f64;
        let z = features.dot(&self.weights) + self.bias;
        // 1/(1+e^-z)
        let predictions = z.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        let error = &predictions - targets;
        
        let weight_gradient = features.t().dot(&error) / n_samples;
        let bias_gradient = error.sum() / n_samples;
        
        self.weights -= &(self.learning_rate * &weight_gradient);
        self.bias -= self.learning_rate * bias_gradient;
    }
    
    fn predict(&self, features: &ArrayView2<f64>) -> Array1<f64> {
        let z = features.dot(&self.weights) + self.bias;
        z.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }
    
    fn predict_one(&self, height: f64, weight: f64, 
                   height_mean: f64, height_std: f64,
                   weight_mean: f64, weight_std: f64) -> f64 {
        // Normalize using training statistics
        let height_norm = (height - height_mean) / height_std;
        let weight_norm = (weight - weight_mean) / weight_std;
        
        // Create input array
        let input = Array2::from_shape_vec((1, 2), vec![height_norm, weight_norm])
            .expect("Failed to create input array");
        
        // Get probability (index 0 because only one sample)
        self.predict(&input.view())[0]
    }
    
    fn predict_class(&self, height: f64, weight: f64,
                    height_mean: f64, height_std: f64,
                    weight_mean: f64, weight_std: f64,
                    threshold: f64) -> &'static str {
        let prob = self.predict_one(height, weight, height_mean, height_std, weight_mean, weight_std);
        
        if prob >= threshold {
            "Female"  // 1.0 = Female
        } else {
            "Male"    // 0.0 = Male
        }
    }
    
    fn accuracy(&self, features: &ArrayView2<f64>, targets: &ArrayView1<f64>) -> f64 {
        let predictions = self.predict(features);
        let threshold = 0.5;
        
        let correct = predictions.iter()
            .zip(targets.iter())
            .filter(|(pred, target)| {
                let pred_class = if **pred >= threshold { 1.0 } else { 0.0 };
                (pred_class - *target).abs() < 0.1
            })
            .count();

        correct as f64 / targets.len() as f64
    }

    fn show_weights(&self) {
        println!("  Weights: height={:.4}, weight={:.4}, bias={:.4}", 
            self.weights[0], self.weights[1], self.bias);
    }
}


pub fn logistic_regression() {
    // Train the model
    let (features, target, height_mean, height_std, weight_mean, weight_std) = match preprocess_data("data/rep_height_weights.csv") {
        Ok(data) => data,
        Err(e) => {
            println!("Error: {:?}", e);
            std::process::exit(1);  
        }
    };
    println!("Loaded {} samples with {} features", features.nrows(), features.ncols());
    
    // Split data
    let split = (features.nrows() as f64 * 0.8) as usize;
    let train_features = features.slice(s![..split, ..]);
    let train_target = target.slice(s![..split]);
    let test_features = features.slice(s![split.., ..]);
    let test_target = target.slice(s![split..]);

    
    // Create and train model
    let mut logistic = Logistic::new(2, 0.1);
    
    // WORKING WITH ARRAY VIEWS TO NOT USE .to_owned() AND MAKE COPIES
    for epoch in 0..1000 {
        logistic.train(&train_features, &train_target);
        
        if epoch % 100 == 0 {
            let train_acc = logistic.accuracy(&train_features, &train_target);
            let test_acc = logistic.accuracy(&test_features, &test_target);
            
            println!("Epoch {}:", epoch);
            logistic.show_weights();
            println!("  Train Accuracy: {:.2}%", train_acc * 100.0);
            println!("  Test Accuracy: {:.2}%", test_acc * 100.0);
        }
    }
    
    // Make predictions on new data
    println!("\n--- Making Predictions ---");
    
    // Example 3: Interactive prediction from user input
    println!("\n--- Enter your own data ---");
    println!("Enter height in cm:");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    let height: f64 = input.trim().parse().unwrap();
    
    println!("Enter weight in kg:");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    let weight: f64 = input.trim().parse().unwrap();
    
    let prob = logistic.predict_one(height, weight, height_mean, height_std, weight_mean, weight_std);
    let class = logistic.predict_class(height, weight, height_mean, height_std, weight_mean, weight_std, 0.5);
    println!("\nPrediction for {}cm, {}kg:", height, weight);
    println!("  Probability of Female: {:.2}%", prob * 100.0);
    println!("  Probability of Male: {:.2}%", (1.0-prob) * 100.0);
    println!("  Predicted class: {}", class);
}
