use crate::matrix::Matrix;
pub mod anova;
use anova::Anova;

pub struct LinearRegressionModel {
    x: Matrix,
    y: Matrix,
    beta_hat: Matrix,
}

impl LinearRegressionModel {
    /// 返回一个空的线性回归模型
    pub fn new() -> Self {
        LinearRegressionModel {
            x: Matrix::new(),
            y: Matrix::new(),
            beta_hat: Matrix::new(),
        }
    }

    /// 线性回归
    /// # params
    /// x: 观测自变量矩阵，输入二维数组\
    /// y: 因变量，输入一维数组
    ///
    /// 使用&self.predict()函数来进行预测
    pub fn fit(&mut self, x: &Vec<Vec<f64>>, y: &Vec<f64>) {
        let mut x_new = x.clone();
        for i in x_new.iter_mut() {
            i.insert(0, 1f64);
        }
        let x_matrix = Matrix::from_2dvec(&x_new).unwrap();
        let mut y_new: Vec<Vec<f64>> = vec![vec![0f64]; y.len()];
        for i in 0..y.len() {
            y_new[i][0] = y[i];
        }
        let y_matrix = Matrix::from_2dvec(&y_new).unwrap();
        self.x = x_matrix;
        self.y = y_matrix;
        match Self::figure_beta_hat(&self.x, &self.y) {
            Ok(bh) => self.beta_hat = bh,
            Err(()) => panic!("get beta hat failed, please check your data"),
        }
    }

    fn figure_beta_hat(x: &Matrix, y: &Matrix) -> Result<Matrix, ()> {
        let mut bh = x
            .transpose()
            .dot(&x)?
            .inverse()?
            .dot(&x.transpose())?
            .dot(&y)?;
        bh.round(8);
        Ok(bh)
    }
    pub fn beta_hat(&self) -> Option<Vec<f64>> {
        if self.beta_hat.row_num == 0 && self.beta_hat.column_num == 0 {
            return None;
        }
        self.beta_hat.get_column(0)
    }

    pub fn predict(&self, to_predict: &Vec<Vec<f64>>) -> Vec<f64> {
        let mut x: Vec<Vec<f64>> = to_predict.clone();
        for i in x.iter_mut() {
            i.insert(0, 1f64);
        }
        let x = Matrix::from_2dvec(&x).unwrap();
        let ans: Vec<Vec<f64>>;
        ans = x.dot(&self.beta_hat).unwrap().to_vec();
        ans.get(0).unwrap().to_vec()
    }

    pub fn r_squared(&self) -> f64 {
        self.ssr() / self.sst()
    }
}

impl Anova for LinearRegressionModel {
    fn sst(&self) -> f64 {
        self.ssr() + self.sse()
    }

    fn sse(&self) -> f64 {
        self.y
            .transpose()
            .dot(&self.y)
            .unwrap()
            .add(
                &self
                    .beta_hat
                    .transpose()
                    .dot(&self.x.transpose())
                    .unwrap()
                    .dot(&self.y)
                    .unwrap()
                    .multiply(-1f64),
            )
            .unwrap()
            .get(0, 0)
            .unwrap()
    }

    ///
    fn ssr(&self) -> f64 {
        let len = self.y.size();
        let mut v = Vec::new();
        for _ in 0..len {
            v.push(1f64);
        }
        let u = Matrix::from_vec(&v);
        self.beta_hat
            .transpose()
            .dot(&self.x.transpose())
            .unwrap()
            .dot(&self.y)
            .unwrap()
            .get(0, 0)
            .unwrap()
            - 1f64 / self.y.size() as f64
                * self
                    .y
                    .transpose()
                    .dot(&u.transpose())
                    .unwrap()
                    .dot(&u)
                    .unwrap()
                    .dot(&self.y)
                    .unwrap()
                    .get(0, 0)
                    .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::LinearRegressionModel;
    use crate::matrix::Matrix;

    #[test]
    fn beta_hat_and_predict() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
        let y = vec![10.0, 20.0, 40.0, 80.0, 100.0];
        let mut linear_model = LinearRegressionModel::new();
        linear_model.fit(&x, &y);
        assert_eq!(
            linear_model.beta_hat,
            Matrix::from_2dvec(&vec![vec![-22.0], vec![24.0],]).unwrap()
        );
        let x2 = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
        let y2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        linear_model.fit(&x2, &y2);
        assert_eq!(
            linear_model.beta_hat,
            Matrix::from_2dvec(&vec![vec![0.0], vec![1.0],]).unwrap()
        );
        let predict = linear_model.predict(&vec![vec![7.0]]);
        assert_eq!(predict, vec![7.0]);
    }

    #[test]
    fn anova() {
        let temp = vec![18.0, 25.0, 45.0, 60.0, 62.0, 75.0, 88.0, 92.0, 99.0, 98.0];
        let mut x: Vec<Vec<f64>> = Vec::new();
        for i in temp {
            x.push(Vec::from(vec![i]));
        }
        let y = vec![15.0, 20.0, 30.0, 40.0, 42.0, 53.0, 60.0, 65.0, 70.0, 78.0];
        let mut model = LinearRegressionModel::new();
        model.fit(&x, &y);
        assert_eq!(
            model.beta_hat.get_column(0).unwrap(),
            vec![-0.20887175, 0.71765667]
        );
        println!("{}", model.r_squared());
        assert!(1e-8 > model.r_squared() - 0.975670026211267);
    }
}
