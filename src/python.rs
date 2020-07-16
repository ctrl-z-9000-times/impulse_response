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
    km: crate::SparseModel,
}

#[pymethods]
impl SparseModel {
    #[new]
    fn new(delta_time: f64, accuracy: f64) -> SparseModel {
        SparseModel::new(delta_time, accuracy)
    }

    fn len(&self) -> usize {
        self.km.len()
    }

    fn touch(&mut self, point: usize) {
        self.km.touch(point)
    }

    fn advance(&mut self, state: Vec<f64>, derivative: &PyAny) -> Vec<f64> {
        // TODO: Better error checking for python land exceptions!
        let mut next_state = vec![0.0; state.len()];
        let len = self.len();
        assert!(derivative.is_callable());
        let deriv_wrapper = |state: crate::SparseVector| -> crate::SparseVector {
            let s_wrapper = SparseVector {
                inner: Some(Arc::new(state)),
            };
            let mut d_wrapper = SparseVector {
                inner: Some(Arc::new(crate::SparseVector::new(len))),
            };
            derivative.call1((s_wrapper, d_wrapper.clone())).unwrap();
            let mut d = None;
            std::mem::swap(&mut d_wrapper.inner, &mut d);
            let d = d.unwrap();
            return std::sync::Arc::<crate::SparseVector>::try_unwrap(d).unwrap();
        };
        self.km.apply(&state, &mut next_state, deriv_wrapper);
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
    inner: Option<std::sync::Arc<RefCell<crate::SparseVector>>>,
}

#[pymethods]
impl SparseVector {
    #[getter]
    fn nonzero(&self) -> PyResult<PyObject> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        Ok(numpy::PyArray1::from_slice(py, &self.inner.as_ref().unwrap().nonzero).to_object(py))
    }
}

#[pyproto]
impl pyo3::PyMappingProtocol for SparseVector {
    fn __getitem__(&self, index: usize) -> PyResult<f64> {
        Ok(self.inner.as_ref().unwrap().dense[index])
    }

    fn __setitem__(&mut self, index: usize, value: f64) -> PyResult<()> {
        let mem = self.inner.as_ref().unwrap().dense[index];
        if mem == 0.0 && value != 0.0 {
            self.inner.as_ref().unwrap().nonzero.push(index);
        } else if mem != 0.0 && value == 0.0 {
            self.inner.as_ref().unwrap().nonzero.remove(index);
        }
        self.inner.as_mut().unwrap().dense[index] = value;
        Ok(())
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.inner.as_ref().unwrap().dense.len())
    }

    fn __delitem__(&mut self, index: usize) -> PyResult<()> {
        self.__setitem__(index, 0.0);
        Ok(())
    }
}
