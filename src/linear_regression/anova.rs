pub trait Anova {
    fn sst(&self) -> f64;

    fn ssr(&self) -> f64;

    fn sse(&self) -> f64;
}
