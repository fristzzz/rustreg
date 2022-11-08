# 构建最小二乘法——使用rust语言


## 内容摘要

线性回归模型是一种在统计学、计量经济学、大数据分析、机器学习等领域常用的基础统计模型。使用rust语言构建了基础的线性回归模型，主要为最小二乘法、方差分析。适用于一元、多元线性回归。代码存储在github:[https://github.com/fristzzz/rustreg]

## 关键词

rust语言，线性回归，最小二乘法

## 构建矩阵计算

最小二乘法通过计算模型$Y=X\beta+u$ 的最小残差平方和得到回归量系数$X$的估计值$\hat{\beta}$，
其中：
$$Y=\begin{Bmatrix}
y_1\\
y_2\\
\vdots\\
y_n\\
\end{Bmatrix},
X=\begin{Bmatrix}
1&x_{11}&x_{12}&\cdots&x_{1m}\\
1&x_{21}&x_{22}&\cdots&x_{2m}\\
\vdots&\vdots&\vdots&\ddots&\vdots\\
1&x_{n1}&x_{n2}&\cdots&x_{nm}\\
\end{Bmatrix},
\beta=\begin{Bmatrix}
\beta_0\\
\beta_1\\
\vdots\\
\beta_m\\
\end{Bmatrix}$$
u为未观测变量，在最小二乘法中被忽略。

则观测值的预测值：
$$\hat{y_i}=\hat{\beta}_0+\hat{\beta}_1x_{i1}+\hat{\beta}_2x_{i2}+\dots+\hat{\beta}_mx_{im}$$

$\hat{\beta}$的计算结果可写为
$$\hat{\boldsymbol\beta}=(\mathbf{X^{T}X)^{-1}X^{T}y}$$

因此需要矩阵计算，为了可复用性，构建基础的矩阵结构体,使用包含一维数组、行数和列数的结构体来代表一个矩阵

```rust
#[derive(Debug, Clone)]
pub struct Matrix {
    elems: Vec<f64>,
    pub row_num: usize,
    pub column_num: usize,
}
```

并实现基础矩阵算法，包括矩阵的加法、点乘、转置、求逆。
1. 矩阵加法
```rust
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
```
2. 矩阵点乘
```rust
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
```
3. 矩阵转置
```rust
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
```

4. 矩阵求逆
```rust
    pub fn inverse(&self) -> Result<Self, ()> {
        let mut res;
        match self.adjoint() { //伴随矩阵,fn adjoint(&self) -> Result<Self, ()>
            Ok(a) => res = a,
            Err(()) => {
                println!("adjoint failed");
                return Err(());
            }
        }
        let det;
        match self.det() { // 矩阵的行列式,fn det(&self) -> Result<f64, ()>
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
```

至此，完成了基本矩阵算法的构建，利用矩阵算法可以方便地计算出最小二乘法的系数估计值$\hat{\beta}$

## 最小二乘法求$\hat{\beta}$

使用单利模式构建线性回归模型，包含观测值向量Y，回归量矩阵X，和系数预测值向量$\hat{\beta}$

```rust
pub struct LinearRegressionModel {
    x: Matrix,
    y: Matrix,
    beta_hat: Matrix,
}
```

利用fit函数输入观测值向量Y和回归量矩阵X，其中Y的数据类型为64位浮点数数组，X的数据类型为64位浮点数二维数组。fit函数将计算出线性回归模型中的$\hat{\beta}$。fit函数代码如下：
```rust
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
```
其中，figure_beta_hat函数利用前文构建的矩阵模型以及$\hat{\beta}$的计算公式$(\mathbf{X^{T}X)^{-1}X^{T}y}$计算得到$\hat{\beta}$

至此，代码实现了系数预测值向量$\hat{\beta}$的计算

## 线性回归预测

给定一组数据$x_1, x_2, \dots, x_m$，利用已得到的$\hat{\beta}$来计算预测值$\hat{y}$, 即：
1. 一元线性回归模型中
$$\hat{y}=(1, x_1, x_2, \dots, x_m)\cdot (\beta_0, \beta_1, \dots, \beta_m) $$
2. 多元线性回归模型中
$$Y=X\beta$$


## 方差分析
在方差分析（ANOVA）中，总平方和（SST）被分成两个部分：残差平方和（SSE）以及回归平方和（SSR）
总平方和SST (sum of squares for total) 是：

$${\displaystyle {\text{SST}}=\sum _{i=1}^{n}(y_{i}-{\bar {y}})^{2}}{\text{SST}}=\sum _{{i=1}}^{n}(y_{i}-{\bar  y})^{2}$$
其中：　
$${\displaystyle {\bar {y}}={\frac {1}{n}}\sum _{i}y_{i}}{\bar  y}={\frac  {1}{n}}\sum _{i}y_{i}$$
同等地：

$${\displaystyle {\text{SST}}=\sum _{i=1}^{n}y_{i}^{2}-{\frac {1}{n}}\left(\sum _{i}y_{i}\right)^{2}}{\text{SST}}=\sum _{{i=1}}^{n}y_{i}^{2}-{\frac  {1}{n}}\left(\sum _{i}y_{i}\right)^{2}$$
回归平方和SSReg (sum of squares for regression。也可写做模型平方和，SSM，sum of squares for model) 是：

$${\displaystyle {\text{SSReg}}=\sum \left({\hat {y}}_{i}-{\bar {y}}\right)^{2}={\hat {\boldsymbol {\beta }}}^{T}\mathbf {X} ^{T}\mathbf {y} -{\frac {1}{n}}\left(\mathbf {y^{T}uu^{T}y} \right),}$$

利用矩阵算法，构建计算回归平方和的函数：
```rust
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
```

残差平方和SSE (sum of squares for error) 是：

$$\displaystyle {\text{SSE}=\sum _{i}\left(y_{i}-\hat {y}_{i}\right)^{2}=\mathbf {y^{T}y-\hat {\boldsymbol {\beta }}^{T}X^{T}y} .}$$

利用矩阵计算写出计算残差平方和的函数：
```rust
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
```

总平方和SST又可写做SSReg和SSE的和：

$${\displaystyle {\text{SST}}=\sum _{i}\left(y_{i}-{\bar {y}}\right)^{2}=\mathbf {y^{T}y} -{\frac {1}{n}}\left(\mathbf {y^{T}uu^{T}y} \right)={\text{SSReg}}+{\text{SSE}}.}$$

则总平方和函数如下：
```rust
    fn sst(&self) -> f64 {
        self.ssr() + self.sse()
    }
```

回归系数R2是：

$${\displaystyle R^{2}={\frac {\text{SSReg}}{\text{SST}}}=1-{\frac {\text{SSE}}{\text{SST}}}.}R^{2}={\frac  {{\text{SSReg}}}{{{\text{SST}}}}}=1-{\frac  {{\text{SSE}}}{{\text{SST}}}}.$$

根据公式构建计算回归系数函数：
```rust
    pub fn r_square(&self) -> f64 {
        self.ssr() / self.sst()
    }
```

## 测试

> 测试文档: hprice1
模型： 
$$price=\beta_0+\beta_1lotsize+\beta_2sqrft+\beta_3bdrms+u $$

测试代码
```rust
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
```
测试通过，并且$\hat{\beta},R^2$结果和stata分析一致

## 总结

本文使用rust语言构建了基础矩阵计算，并实现了线性回归模型中的最小二乘法和方差分析。经过测试与stata分析得出的结果一致。

## 参考文献
[1]维基百科.线性回归[OL]https://zh.m.wikipedia.org/zh-hans/%E7%B7%9A%E6%80%A7%E5%9B%9E%E6%AD%B8
