use std::fmt;

#[derive(Debug)]
pub struct Matrix {
    elems: Vec<f64>,
    pub row_num: usize,
    pub column_num: usize,
}

//格式化输出矩阵，便于查看和调试
impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = String::new();
        for i in 0..self.row_num {
            if i == 0 {
                s.push('[');

                s.push('\n');
            }
            for j in 0..self.column_num {
                if j == 0 {
                    s.push('|');
                }
                s.push_str(&format!(
                    "{:>8}|",
                    &self.elems[i * self.column_num + j].to_string()
                ));
                s.push_str("  ");
                if j == self.column_num - 1 {
                    s.push_str("\n");
                }
            }
            if i == self.row_num - 1 {
                s.push(']');
            }
        }
        write!(f, "{}", s)
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        (self.row_num == other.row_num)
            && (self.column_num == other.column_num)
            && self.elems.eq(&other.elems)
    }
}
impl Eq for Matrix {}
//矩阵类型
impl Matrix {
    pub fn new() -> Self {
        Matrix {
            elems: Vec::new(),
            row_num: 0,
            column_num: 0,
        }
    }

    ///由二维数组创建矩阵
    ///# Errors
    /// 当矩阵每行元素数量不相等时，返回错误
    pub fn from_2dvec(v: &Vec<Vec<f64>>) -> Result<Self, ()> {
        if v.is_empty() {
            return Ok(Self::new());
        }
        let mut elems = vec![];
        let len = v[0].len();
        for i in v.iter() {
            if i.len() != len {
                return Err(());
            }
            for &j in i.iter() {
                elems.push(j);
            }
        }
        let row_num: usize = v.len();
        let column_num: usize = v[0].len();
        Ok(Self {
            column_num,
            row_num,
            elems,
        })
    }

    pub fn from_vec(v: &Vec<f64>) -> Self {
        Self {
            column_num: v.len(),
            row_num: 1,
            elems: v.clone(),
        }
    }

    pub fn rows(&self) -> Vec<Vec<f64>> {
        self.to_vec()
    }

    pub fn get(&self, index_i: usize, index_j: usize) -> Option<f64> {
        match self.elems.get(index_i + index_j * self.column_num) {
            Some(&x) => Some(x),
            None => None,
        }
    }

    ///矩阵点乘，参数为右侧矩阵
    /// # Errors
    /// 当前矩阵的列数与右侧矩阵的行数不相等，
    /// 即无法相乘时返回错误的空矩阵
    pub fn dot(&self, right: &Self) -> Result<Self, ()> {
        if self.column_num != right.row_num {
            return Err(());
        }
        let row_num = self.row_num;
        let column_num = right.column_num;
        let mut elems = Vec::new();
        for i in 0..self.row_num {
            for j in 0..right.column_num {
                elems.push(0.0);
                for k in 0..self.column_num {
                    elems[i * column_num + j] +=
                        self.elems[i * self.column_num + k] * right.elems[j + k * right.column_num];
                }
            }
        }

        Ok(Self {
            row_num,
            column_num,
            elems,
        })
    }

    pub fn add(&mut self, right: &Self) -> Result<Self, ()> {
        if self.row_num != right.row_num || self.column_num != right.column_num {
            return Err(());
        }
        let mut res = Self::new();
        for i in 0..self.row_num {
            for j in 0..self.column_num {
                res.elems.push(
                    self.elems[j + i * self.column_num] + right.elems[j + i * self.column_num],
                );
            }
        }
        res.row_num = self.row_num;
        res.column_num = self.column_num;
        Ok(res)
    }

    pub fn multiply(&self, plier: f64) -> Self {
        let mut res = Self::new();
        for i in self.elems.iter() {
            res.elems.push(i * plier);
        }
        res.row_num = self.row_num;
        res.column_num = self.column_num;
        res
    }

    ///矩阵转置
    pub fn transpose(&self) -> Self {
        let row_num = self.column_num;
        let column_num = self.row_num;
        let mut elems = Vec::new();
        for i in 0..self.column_num {
            for j in 0..self.row_num {
                elems.push(self.elems[j * self.column_num + i]);
            }
        }
        Self {
            column_num,
            row_num,
            elems,
        }
    }

    ///矩阵求逆
    /// # Errors
    /// when det equal 0, inverse will return an error: Err(())
    pub fn inverse(&self) -> Result<Self, ()> {
        let mut res;
        match self.adjoint() {
            Ok(a) => res = a,
            Err(()) => {
                println!("adjoint failed");
                return Err(());
            }
        }
        let det;
        match self.det() {
            Ok(d) => det = d,
            Err(()) => {
                println!("det failed");
                return Err(());
            }
        }
        if det == 0f64 {
            return Err(());
        }
        for elem in res.elems.iter_mut() {
            *elem = *elem / det;
        }
        Ok(res)
    }

    pub fn adjoint(&self) -> Result<Self, ()> {
        if !self.is_square_matrix() {
            return Err(());
        }
        let mut elems = Vec::new();
        for i in 0..self.row_num {
            for j in 0..self.column_num {
                elems.push(self.cofactor(i, j).unwrap());
            }
        }
        Ok(Self { elems, ..*self }.transpose())
    }

    pub fn cofactor(&self, x: usize, y: usize) -> Result<f64, ()> {
        if !self.is_square_matrix() {
            return Err(());
        }
        let mut sub_matrix = Vec::new();
        for i in 0..self.row_num {
            for j in 0..self.column_num {
                if i == x || j == y {
                    continue;
                }
                sub_matrix.push(self.elems[i * self.column_num + j]);
            }
        }
        Ok((-1i32).pow((x + y) as u32) as f64 * Self::det_vec(&sub_matrix, self.row_num - 1))
    }

    pub fn det(&self) -> Result<f64, ()> {
        if !self.is_square_matrix() {
            return Err(());
        }
        Ok(Self::det_vec(&self.elems, self.row_num))
    }
    fn det_vec(v: &Vec<f64>, size: usize) -> f64 {
        if size == 1 {
            return v[0];
        }
        let mut sum = 0f64;
        for i in 0..size {
            let mut new_v = Vec::new();
            for row in 0..size {
                for column in 1..size {
                    if row == i {
                        continue;
                    }
                    new_v.push(v[row * size + column]);
                }
            }
            sum += (-1i32).pow(i as u32) as f64 * v[i * size] * Self::det_vec(&new_v, size - 1);
        }

        sum
    }
    ///将矩阵转化为二维数组
    pub fn to_vec(&self) -> Vec<Vec<f64>> {
        let mut ans = Vec::new();
        let _ = self
            .elems
            .chunks(self.column_num)
            .map(|row| {
                ans.push(row.to_vec());
            })
            .collect::<Vec<_>>();
        ans
    }

    pub fn is_square_matrix(&self) -> bool {
        self.row_num == self.column_num
    }

    pub fn round(&mut self, place: u32) {
        for elem in self.elems.iter_mut() {
            *elem = (*elem * 10_i32.pow(place) as f64).round() / 10_i32.pow(place) as f64;
        }
    }

    pub fn size(&self) -> usize {
        self.elems.len()
    }

    pub fn remove_row(&self, row_index: usize) -> Option<Self> {
        if !row_index < self.row_num {
            return None;
        }

        let mut elems = Vec::new();
        for i in 0..self.row_num {
            if i == row_index {
                continue;
            }
            for j in 0..self.column_num {
                elems.push(self.elems[j + i * self.column_num]);
            }
        }
        Some(Self {
            elems,
            row_num: self.row_num - 1,
            column_num: self.column_num,
        })
    }
    pub fn remove_column(&self, column_index: usize) -> Option<Self> {
        if !column_index < self.column_num {
            return None;
        }

        let mut elems = Vec::new();
        for i in 0..self.row_num {
            for j in 0..self.column_num {
                if j == column_index {
                    continue;
                }
                elems.push(self.elems[j + i * self.column_num]);
            }
        }
        Some(Self {
            elems,
            row_num: self.row_num,
            column_num: self.column_num - 1,
        })
    }
    pub fn get_column(&self, column_index: usize) -> Option<Vec<f64>> {
        if !column_index < self.column_num || self.elems.is_empty() {
            return None;
        }
        let mut result = Vec::new();
        for i in 0..self.row_num {
            result.push(self.elems[self.column_num * i + column_index]);
        }
        Some(result)
    }
    pub fn get_row(&self, row_index: usize) -> Option<Vec<f64>> {
        if !row_index < self.row_num || self.elems.is_empty() {
            return None;
        }
        let mut result = Vec::new();
        for i in 0..self.column_num {
            result.push(self.elems[row_index * self.column_num + i]);
        }

        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;
    #[test]
    fn vec_to_matrix() {
        let v2: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0]];
        assert!(Matrix::from_2dvec(&v2).is_err());

        let v: Vec<f64> = vec![1.0, 2.0];
        let m = Matrix::from_vec(&v);
        assert_eq!(m.column_num, 2);
        assert_eq!(m.row_num, 1);
        assert_eq!(m.to_vec(), vec![vec![1.0, 2.0]]);
    }

    #[test]
    fn dot() {
        let v1: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let v2: Vec<Vec<f64>> = vec![vec![1.0, 2.0], vec![4.0, 5.0], vec![7.0, 8.0]];
        let v3: Vec<Vec<f64>> = vec![
            vec![1.0, 4.0],
            vec![1.0, 6.0],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
        ];

        let m1 = Matrix::from_2dvec(&v1).unwrap();
        let m2 = Matrix::from_2dvec(&v2).unwrap();
        let m3 = Matrix::from_2dvec(&v3).unwrap();
        assert!(m1.dot(&m3).is_err());
        let res = m1.dot(&m2).unwrap();
        assert_eq!(res.to_vec(), vec![vec![30.0, 36.0], vec![66.0, 81.0]]);

        let v4 = vec![vec![1.0, 2.0], vec![3.0, 5.0], vec![5.0, 4.0]];
        let v5 = vec![vec![6.0, 8.0, 3.0], vec![9.0, 9.0, 7.0]];

        assert_eq!(
            Matrix::from_2dvec(&v4)
                .unwrap()
                .dot(&Matrix::from_2dvec(&v5).unwrap())
                .unwrap()
                .to_vec(),
            vec![
                vec![24.0, 26.0, 17.0],
                vec![63.0, 69.0, 44.0],
                vec![66.0, 76.0, 43.0],
            ]
        );
    }

    #[test]
    fn to_vec() {
        let v: Vec<Vec<f64>> = vec![
            vec![1.0, 4.0],
            vec![1.0, 6.0],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
        ];
        let m = Matrix::from_2dvec(&v).unwrap();
        assert_eq!(m.to_vec(), v);
    }

    #[test]
    fn transpose() {
        let v: Vec<Vec<f64>> = vec![vec![1.0, 2.0], vec![4.0, 5.0], vec![7.0, 8.0]];
        let m = Matrix::from_2dvec(&v).unwrap();
        let m = m.transpose();
        assert_eq!(m.to_vec(), vec![vec![1.0, 4.0, 7.0], vec![2.0, 5.0, 8.0]]);
        assert_eq!(
            Matrix::from_2dvec(&vec![vec![1.0,],])
                .unwrap()
                .transpose()
                .to_vec(),
            vec![vec![1.0,],]
        );
    }

    #[test]
    fn det() {
        let v = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let m = Matrix::from_2dvec(&v);
        assert!(!m.is_err());
        let m = m.unwrap();
        assert_eq!(m.det().unwrap(), -2f64);

        let v2 = vec![
            vec![2.0, 4.0, 4.0],
            vec![7.0, 6.0, 5.0],
            vec![3.0, 1.0, 8.0],
        ];
        let m2 = Matrix::from_2dvec(&v2).unwrap();
        assert_eq!(m2.det().unwrap(), -122f64);

        let v3 = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 10.0],
        ];
        let m3 = Matrix::from_2dvec(&v3).unwrap();
        assert_eq!(m3.det().unwrap(), -3f64);

        assert_eq!(
            Matrix::from_2dvec(&vec![
                vec![5.0, 7.0, 5.0, 8.0],
                vec![5.0, 7.0, 4.0, 9.0],
                vec![6.0, 7.0, 3.0, 4.0],
                vec![2.0, 8.0, 9.0, 6.0],
            ])
            .unwrap()
            .det()
            .unwrap(),
            155f64
        );
    }

    #[test]
    fn cofactor() {
        let v = vec![
            vec![2.0, 4.0, 4.0],
            vec![7.0, 6.0, 5.0],
            vec![3.0, 1.0, 8.0],
        ];
        let m = Matrix::from_2dvec(&v).unwrap();
        assert_eq!(m.cofactor(0, 0).unwrap(), 43f64);
        assert_eq!(m.cofactor(1, 0).unwrap(), -28f64);
    }

    #[test]
    fn adjoint() {
        let v = vec![
            vec![2.0, 4.0, 4.0],
            vec![7.0, 6.0, 5.0],
            vec![3.0, 1.0, 8.0],
        ];
        let m = Matrix::from_2dvec(&v).unwrap();
        let res = m.adjoint().unwrap();
        assert_eq!(
            res.to_vec(),
            vec![
                vec![43.0, -28.0, -4.0],
                vec![-41.0, 4.0, 18.0],
                vec![-11.0, 10.0, -16.0],
            ]
        )
    }

    #[test]
    fn inverse() {
        let v = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        // let v1 = vec![
        //     vec![1.0, 2.0, 3.0],
        //     vec![4.0, 5.0, 6.0],
        //     vec![7.0, 8.0, 10.0],
        // ];
        let m = Matrix::from_2dvec(&v).unwrap();
        // let m1 = Matrix::from_2dvec(&v1).unwrap();
        assert!(m.inverse().is_err());

        let v2 = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 6.0, 5.0, 4.0],
            vec![7.0, 4.0, 6.0, 4.0],
        ];
        let m2 = Matrix::from_2dvec(&v2).unwrap();
        println!("{}", m2.inverse().unwrap());
        assert_eq!(
            vec![
                vec![1.5, -1.0, 0.5, -0.0],
                vec![-2.375, 1.5, -0.375, -0.25],
                vec![-1.75, 1.0, -0.75, 0.5],
                vec![2.375, -1.25, 0.625, -0.25],
            ],
            m2.inverse().unwrap().to_vec()
        );
    }

    #[test]
    fn get() {
        let v = vec![vec![1.0], vec![2.0]];
        let m = Matrix::from_2dvec(&v).unwrap();
        assert_eq!(m.get(0, 0), Some(1.0));
        assert_eq!(m.get(1, 0), Some(2.0));
    }

    #[test]
    fn round() {
        let v = vec![vec![1.293214814123]];
        let mut m = Matrix::from_2dvec(&v).unwrap();
        m.round(2u32);
        let mut res = m.to_vec();
        let res = res[0][0];
        assert_eq!(res, 1.29f64);
    }

    #[test]
    fn get_row() {
        let mut m = Matrix::from_2dvec(&vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let res = m.get_row(0).unwrap();
        assert_eq!(res, vec![1.0, 2.0]);
        let m2 = Matrix::from_2dvec(&vec![vec![]]).unwrap();
        println!("{}", m2.row_num);
        assert!(m2.get_row(0).is_none());
    }

    #[test]
    fn get_column() {
        let mut m = Matrix::from_2dvec(&vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let res = m.get_column(0).unwrap();
        assert_eq!(res, vec![1.0, 3.0]);
        assert_eq!(m.get_column(1).unwrap(), vec![2.0, 4.0]);
        let m2 = Matrix::from_2dvec(&vec![vec![]]).unwrap();
        assert!(m2.get_column(0).is_none());

        let m3 = Matrix::from_2dvec(&vec![
            vec![1.0, 4.0, 5.0, 3.0, 123.3, 12321.0, 123.0],
            vec![2.4, 421.4, 21.4, 421.1, 123.5, 12.9, 123.9],
        ])
        .unwrap();

        print!("{:?}", m3.get_column(6).unwrap());
    }

    #[test]
    fn add() {
        let v: Vec<Vec<f64>> = vec![
            vec![1.0, 4.0],
            vec![1.0, 6.0],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
        ];
        let v1: Vec<Vec<f64>> = vec![
            vec![1.0, 4.0],
            vec![1.0, 6.0],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
        ];
        let v2: Vec<Vec<f64>> = vec![
            vec![1.0, 4.0],
            vec![1.0, 6.0],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
        ];
        let mut m = Matrix::from_2dvec(&v).unwrap();
        let mut m2 = Matrix::from_2dvec(&v1).unwrap();
        assert_eq!(
            m.add(&m2).unwrap().to_vec(),
            vec![
                vec![2.0, 8.0],
                vec![2.0, 12.0],
                vec![2.0, 2.0],
                vec![2.0, 2.0],
            ]
        );
    }

    #[test]
    fn multi() {
        let v = vec![
            vec![1.0, 4.0],
            vec![1.0, 6.0],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
        ];
        let m = Matrix::from_2dvec(&v).unwrap();
        assert_eq!(
            m.multiply(2_f64).to_vec(),
            vec![
                vec![2.0, 8.0],
                vec![2.0, 12.0],
                vec![2.0, 2.0],
                vec![2.0, 2.0],
            ]
        );
    }
}
