use pyo3::prelude::*;
use std::sync::Arc;

#[pymodule]
fn sparse_model(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SparseModel>()?;
    m.add_class::<SparseVector>()?;
    Ok(())
}

#[pyclass]
struct SparseModel {
    m: crate::SparseModel,
}

#[pymethods]
impl SparseModel {
    #[new]
    fn new(delta_time: f64, accuracy: f64) -> SparseModel {
        SparseModel::new(delta_time, accuracy)
    }

    fn len(&self) -> usize {
        self.m.len()
    }

    fn touch(&mut self, point: usize) {
        self.m.touch(point)
    }

    fn advance(&mut self, state: Vec<f64>, derivative: &PyAny) -> Vec<f64> {
        // TODO: Better error checking for python land exceptions!
        let mut next_state = vec![0.0; state.len()];
        let len = self.len();
        assert!(derivative.is_callable());
        self.m.advance(
            &state,
            &mut next_state,
            |state: &crate::SparseVector, deriv: &mut crate::SparseVector| {
                let mut d_wrapper = SparseVector {
                    inner: Some(Arc::new(crate::SparseVector::new(len))),
                };
                derivative.call1((state, d_wrapper.clone())).unwrap();
                let mut d = None;
                std::mem::swap(&mut d_wrapper.inner, &mut d);
                let d = d.unwrap();
                return std::sync::Arc::<crate::SparseVector>::try_unwrap(d).unwrap();
            },
        );
        return next_state;
    }
}

/*
TODO: Design the python API for SparseVector.

Idea:
- It should act like an np.array over the dense, by default.
- magic attribute: nonzero -> py list
*/
#[pyclass]
#[derive(Clone)]
struct SparseVector {
    inner: crate::SparseVector,
}

#[pymethods]
impl SparseVector {
    #[getter]
    fn nonzero(&self) -> Vec<usize> {
        self.inner.nonzero.clone
    }
}

#[pyproto]
impl pyo3::PyMappingProtocol for SparseVector {
    fn __getitem__(&self, index: usize) -> PyResult<f64> {
        Ok(self.inner.data[index])
    }

    fn __setitem__(&mut self, index: usize, value: f64) -> PyResult<()> {
        let mem = self.inner.data[index];
        if mem == 0.0 && value != 0.0 {
            self.inner.nonzero.push(index);
        } else if mem != 0.0 && value == 0.0 {
            self.inner.nonzero.remove(index);
        }
        self.inner.data[index] = value;
        Ok(())
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.inner.data.len())
    }

    fn __delitem__(&mut self, index: usize) -> PyResult<()> {
        self.__setitem__(index, 0.0);
        Ok(())
    }
}
