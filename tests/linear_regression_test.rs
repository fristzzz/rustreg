use csv;
use statistics_paper::linear_regression::LinearRegressionModel;
use statistics_paper::matrix::*;
use std::{fs, io::BufReader};

#[test]
fn insurance_linear_regression() {
    let insurance = fs::File::open("./src/datasets/insurance.csv").unwrap();
    let reader = BufReader::new(insurance);
    let mut r = csv::Reader::from_reader(reader);

    let mut data_train = Vec::new();
    for record in r.records() {
        let mut new_vec: Vec<f64> = Vec::new();
        let record = record.unwrap();
        for elem in &record {
            match elem.parse::<f64>() {
                Ok(x) => new_vec.push(x),
                Err(_) => unsafe {
                    let mut bytes = (*elem.clone()).to_string();
                    let mut bytes = (*bytes.as_bytes_mut()).to_vec();
                    let mut code: f64 = 0f64;
                    for i in (0..bytes.len()).rev() {
                        code += ((bytes[i] as usize * 10usize.pow(i as u32)) as f64)
                            / 10u32.pow(i as u32) as f64;
                    }
                    new_vec.push(code);
                },
            }
        }
        data_train.push(new_vec);
    }
    println!("data train len: {}", data_train.len());
    let mut m = Matrix::from_2dvec(&data_train).unwrap();

    println!("m.row_num: {}", m.row_num);
    println!("m.column_num: {}", m.column_num);
    let mut y = m.get_column(6).unwrap();
    m = m.remove_column(6).unwrap();
    data_train = m.to_vec();
    let mut linear = LinearRegressionModel::new();
    linear.fit(&data_train, &y);
    println!("{:?}", linear.beta_hat().unwrap());
}
