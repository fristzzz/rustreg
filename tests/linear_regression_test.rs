use csv;
use serde;
use serde_derive::{self, Deserialize};
use statistics_paper::linear_regression::LinearRegressionModel;
use statistics_paper::matrix::*;
use std::{fs, io::BufReader};

#[test]
fn reg_insurance() {
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
    let mut m = Matrix::from_2dvec(&data_train).unwrap();

    let mut y = m.get_column(6).unwrap();
    m = m.remove_column(6).unwrap();
    data_train = m.to_vec();
    let mut linear = LinearRegressionModel::new();
    linear.fit(&data_train, &y);
}

#[test]
fn hprice1() {
    use statistics_paper::linear_regression::anova::Anova;
    #[derive(Debug, Deserialize)]
    struct Model {
        price: f64,
        lotsize: f64,
        sqrft: f64,
        bdrms: f64,
    }
    let mut x: Vec<Vec<f64>> = Vec::new();
    let mut y: Vec<f64> = Vec::new();
    let mut datas: Vec<Model> = Vec::new();

    let mut data = csv::Reader::from_path("./src/datasets/hprice1.csv").unwrap();

    for record in data.deserialize() {
        datas.push(record.unwrap());
    }
    for model in datas.iter() {
        x.append(&mut vec![vec![model.lotsize, model.sqrft, model.bdrms]]);
    }
    for model in datas.iter() {
        y.append(&mut vec![model.price]);
    }
    let mut regMod = LinearRegressionModel::new();
    regMod.fit(&x, &y);
    let beta_hat = regMod.beta_hat().unwrap();
    println!("beta hat of hprice1: {:#?}", beta_hat);
    assert_eq!(
        beta_hat,
        vec![-21.7703086, 0.00206771, 0.12277819, 13.85252186,]
    );

    println!("r squared of hprice1: {}", regMod.r_squared());
    assert!(regMod.r_squared() - 0.6724 < 1e-5);
}
