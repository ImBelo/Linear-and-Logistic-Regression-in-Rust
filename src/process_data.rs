use core::f64;
use std::error::Error;
use ndarray::Axis;
use polars::{prelude::*};
use ndarray::{Array, Array1, Array2, stack};
pub fn preprocess_data(path: &str) -> Result<(Array2<f64>, Array1<f64>, f64, f64, f64, f64), Box<dyn Error>> {
    // define the coloumns and their types
    let schema = Schema::from_iter(vec![
        Field::new("id".into(), DataType::Int32),
        Field::new("sex".into(), DataType::String),
        Field::new("weight".into(), DataType::Float64),
        Field::new("height".into(), DataType::Float64),
        Field::new("weight_rep".into(), DataType::Float64),
        Field::new("height_rep".into(), DataType::Float64),
    ]);
    
    // define lazy csv reader based on the definitions of coloumns
    // need to pass an Arc(schema) cause it used multithreading under the hood
    // also transforma the coloumn "sex" to 0 if "M" and 1 if "F"
    let lf = LazyCsvReader::new(path.into())
        .with_schema(Some(Arc::new(schema)))
        .with_has_header(true)
        .finish().unwrap()
        .with_column(
            when(col("sex").eq(lit("M")))
                .then(lit(0.0))
                .otherwise(lit(1.0))
                .alias("sex_numeric")
        )
        .collect().expect("Could not load data into dataframe");
    
    // get coloums for lazy dataframe
    let height = lf.column("height").unwrap().f64()?;
    let weight = lf.column("weight").unwrap().f64()?;
    let sex = lf.column("sex_numeric").unwrap().f64()?;
    let len = height.len();
    
    // transform them into ndarray array
    let height_arr = Array1::from_iter(height.into_iter().map(|n| n.unwrap_or(f64::NAN)));
    let weight_arr = Array1::from_iter(weight.into_iter().map(|n| n.unwrap_or(f64::NAN)));
    let sex_arr = Array::from_iter(sex.into_iter().map(|n| n.unwrap_or(f64::NAN)));

    // trasform the 2 1-coloumn vectors height weight into a 1 2-coloumn array
    // (height,weight)(height1,weight1).....
    let mut features = stack(Axis(1), &[height_arr.view(),weight_arr.view()])?;

    println!("Loaded {} valid samples", len);
    
    // normalize the data cause gradients descent is not scale invariant
    // and also because learning rate is the same for each dimention
    // we could fix this by using different learning rate for each dimention or normalize the data
    // and use the same learning rate
    let height_col = features.column(0);
    let height_mean = height_col.mean().unwrap();
    let height_std = height_col.std(0.0);
    
    let weight_col = features.column(1);
    let weight_mean = weight_col.mean().unwrap();
    let weight_std = weight_col.std(0.0);
    
    println!("Training statistics:");
    println!("  Height: mean={:.2}, std={:.2}", height_mean, height_std);
    println!("  Weight: mean={:.2}, std={:.2}", weight_mean, weight_std);
    
    // normalize features because gradient descent is scale variant
    for col in 0..features.ncols() {
        let mut column = features.column_mut(col);
        let mean = if col == 0 { height_mean } else { weight_mean };
        let std = if col == 0 { height_std } else { weight_std };
        if std > 1e-10 {
            column.mapv_inplace(|x| (x - mean) / std);
        }
    }
    
    let target = Array1::from_iter(sex_arr);
    
    Ok((features, target, height_mean, height_std, weight_mean, weight_std))
}

