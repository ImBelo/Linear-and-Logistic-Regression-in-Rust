use ndarray::{Array2, Array1};

struct LinearRegression {
    weights: Array1<f64>,
    learning_rate: f64,
}

impl LinearRegression {
    pub fn new(n_features: usize, learning_rate: f64) -> Self {
        let weights = Array1::zeros(n_features);  // Initialize weights to zeros
        Self { weights, learning_rate }
    }
    
    pub fn train(&mut self, input_features: &Array2<f64>, target_features: &Array1<f64>) {
        //(x1,x2,x3)()
        let h = input_features.dot(&self.weights);   // (n_samples,)
        let error = &h - target_features;            // (n_samples,)
        
        let gradient = 2.0 * input_features.t().dot(&error) / input_features.nrows() as f64;
        
        self.weights = &self.weights - self.learning_rate * &gradient;
    }
    
    pub fn show_weights(&self) {
        println!("Weights: {:?}", self.weights);
    }
}
use ndarray::Axis;

#[test]
fn linear() {
    
    let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
    let y = Array1::from_vec(vec![3.0, 5.0, 7.0]);
    let ones = Array2::ones((x.nrows(), 1));
    let x_with_bias = ndarray::concatenate(Axis(1), &[ones.view(), x.view()]).unwrap();
    println!("x_with_bias shape: {:?}", x_with_bias.shape());  // [3, 2]
    
    let mut model = LinearRegression::new(x_with_bias.shape()[1], 0.1); 
    
    // Train for 1000 epochs
    for epoch in 0..1000 {
        model.train(&x_with_bias, &y);
        
        if epoch % 100 == 0 {
            model.show_weights();
        }
    }
    
    // Test prediction
    let predictions = model.predict(&x_with_bias);
    println!("Predictions: {:?}", predictions);
    assert!((predictions[0] - 3.0).abs() < 0.1);
    assert!((predictions[1] - 5.0).abs() < 0.1);
    assert!((predictions[2] - 7.0).abs() < 0.1);
}

impl LinearRegression {
    pub fn predict(&self, input_features: &Array2<f64>) -> Array1<f64> {
        input_features.dot(&self.weights)
    }
}

