/** Container for a vector which tracks its non-zero elements. */
#[derive(Debug)]
pub struct SparseVector {
    /** Dense array of all elements in the vector, including any zeros.

    The user is responsible for updating the `nonzero` list! Check data elements
    before overwriting them, and if you change a zero to a non-zero value then
    append the index to the `nonzero` list. */
    pub data: Vec<f64>,

    /** Indices of the non-zero elements in `data`.

    - May be unsorted,
    - May contain duplicates,
    - May refer to elements with a value of zero.
    */
    pub nonzero: Vec<usize>,
}

impl SparseVector {
    #[doc(hidden)]
    pub fn new(size: usize) -> Self {
        SparseVector {
            data: vec![0.0; size],
            nonzero: vec![],
        }
    }

    #[doc(hidden)]
    pub fn clear(&mut self) {
        for idx in &self.nonzero {
            self.data[*idx] = 0.0;
        }
        self.nonzero.clear();
    }

    /// Cleans up the nonzero list
    ///
    /// - Sorts and deduplicates,
    /// - Removes zero elements,
    /// - Checks for the inclusion of all non-zero elements.
    #[doc(hidden)]
    pub fn clean(&mut self) {
        self.nonzero.sort_unstable();
        self.nonzero.dedup();
        let data = &self.data;
        self.nonzero.retain(|idx| data[*idx] != 0.0);
        // Check that nonzero contains *all* non-zero elements.
        debug_assert!(self
            .data
            .iter()
            .enumerate()
            .all(|(i, v)| *v == 0.0 || self.nonzero.binary_search(&i).is_ok()));
    }

    /// Performs the equation: `A*x + B => B`
    /// Where A & B are SparseVectors and x is a scalar.
    #[doc(hidden)]
    pub fn add_multiply(&mut self, a: &SparseVector, x: f64) {
        debug_assert_eq!(a.data.len(), self.data.len());
        for point in &a.nonzero {
            let value = &mut self.data[*point];
            if *value == 0.0 {
                self.nonzero.push(*point);
            }
            *value += a.data[*point] * x;
        }
    }

    #[doc(hidden)]
    pub fn max_abs_diff(a: &SparseVector, b: &SparseVector) -> f64 {
        debug_assert_eq!(a.data.len(), b.data.len());
        let mut max: f64 = 0.0;
        for idx in &a.nonzero {
            max = max.max((a.data[*idx] - b.data[*idx]).abs());
        }
        for idx in &b.nonzero {
            max = max.max((a.data[*idx] - b.data[*idx]).abs());
        }
        return max;
    }
}

impl Clone for SparseVector {
    fn clone(&self) -> Self {
        let mut x = SparseVector::new(self.data.len());
        x.nonzero = self.nonzero.clone();
        for idx in &self.nonzero {
            x.data[*idx] = self.data[*idx];
        }
        return x;
    }
}

#[derive(Debug)]
struct SparseCoordinate {
    row: usize,
    column: usize,
    value: f64,
}

/// Compressed Sparse Row Matrix.
#[derive(Debug)]
pub struct SparseMatrix {
    pub data: Vec<f64>,
    pub row_ranges: Vec<usize>,
    pub column_indices: Vec<usize>,
}

impl Default for SparseMatrix {
    fn default() -> SparseMatrix {
        SparseMatrix {
            data: vec![],
            column_indices: vec![],
            row_ranges: vec![0],
        }
    }
}

impl SparseMatrix {
    pub fn len(&self) -> usize {
        self.row_ranges.len() - 1
    }

    pub fn resize(&mut self, new_size: usize) {
        assert!(new_size >= self.len()); // SparseMatrix can not shrink, can only expand.
        self.row_ranges.resize(new_size + 1, self.data.len());
    }

    pub fn write_columns(&mut self, columns: &[usize], rows: &[SparseVector]) {
        let mut delete_columns = vec![false; self.len()];
        for c in columns {
            delete_columns[*c] = true;
        }
        let mut coords = Vec::with_capacity(rows.iter().map(|sv| sv.nonzero.len()).sum());
        for (c_idx, row) in columns.iter().zip(rows) {
            for r_idx in &row.nonzero {
                coords.push(SparseCoordinate {
                    row: *r_idx,
                    column: *c_idx,
                    value: row.data[*r_idx],
                });
            }
        }
        coords.sort_unstable_by(|a, b| a.row.cmp(&b.row));
        let mut insert_iter = coords.iter().peekable();
        let mut result = SparseMatrix::default();
        let max_new_len = self.data.len() + coords.len();
        result.data.reserve(max_new_len);
        result.column_indices.reserve(max_new_len);
        result.row_ranges.reserve(self.row_ranges.len());
        for (row, (row_start, row_end)) in self
            .row_ranges
            .iter()
            .zip(self.row_ranges.iter().skip(1))
            .enumerate()
        {
            // Filter out the existing data from all of the columns which are
            // being written to.
            for index in *row_start..*row_end {
                let column = self.column_indices[index];
                if !delete_columns[column] {
                    result.data.push(self.data[index]);
                    result.column_indices.push(column);
                }
            }
            // Write the new data for the columns.
            while insert_iter.peek().is_some() && insert_iter.peek().unwrap().row == row {
                let coord = insert_iter.next().unwrap();
                result.data.push(coord.value);
                result.column_indices.push(coord.column);
            }
            result.row_ranges.push(result.data.len());
        }
        std::mem::swap(self, &mut result);
    }

    /// Matrix * Vector Multiplication.
    ///
    /// Computes: `self * src => dst`.
    /// Arguments src & dst are dense column vectors.
    pub fn x_vector(&self, src: &[f64], dst: &mut [f64]) {
        assert!(src.len() == self.len());
        assert!(dst.len() == self.len());
        for (row, (row_start, row_end)) in self
            .row_ranges
            .iter()
            .zip(self.row_ranges.iter().skip(1))
            .enumerate()
        {
            // dst[row] = 0.0;
            // for index in *row_start..*row_end {
            //     let column = self.column_indices[index];
            //     let weight = self.data[index];
            //     dst[row] += weight * src[column];
            // }
            dst[row] = (*row_start..*row_end)
                .map(|index| self.data[index] * src[self.column_indices[index]])
                .sum();
        }
    }
}
