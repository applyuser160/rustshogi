use super::common;

use pyo3::prelude::*;

#[derive(Default, PartialEq, Debug, Clone)]
#[pyclass]
pub struct Address {
    pub value: u8,
}

impl Address {
    pub fn from_number(index: u8) -> Self {
        Self { value: index }
    }

    pub fn from_numbers(c: u8, r: u8) -> Self {
        Self { value: r * 11 + c }
    }

    pub fn from_string(string: &str) -> Self {
        Self {
            value: (string.as_bytes()[0] - 48) * 11 + (string.as_bytes()[1] - 96),
        }
    }

    pub fn get_column(&self) -> u8 {
        self.value % 11
    }

    pub fn get_row(&self) -> u8 {
        self.value / 11
    }

    pub fn to_index(&self) -> u8 {
        self.value
    }

    pub fn to_string(&self) -> String {
        let mut string: String = String::new();
        let column: char = common::integer_to_char(self.value % 11);
        let mut row: char = common::integer_to_char(self.value / 11);
        row = common::number_to_alphabet(row);
        string.push(column);
        string.push(row);
        string
    }
}

#[pymethods]
impl Address {
    #[new]
    pub fn python_new(column: u8, row: u8) -> Self {
        Self {
            value: row * 11 + column,
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Address(column={}, row={})",
            self.get_column(),
            self.get_row()
        )
    }

    pub fn __str__(&self) -> String {
        format!(
            "Address(column={}, row={})",
            self.get_column(),
            self.get_row()
        )
    }

    pub fn __eq__(&self, other: &Self) -> bool {
        self.value == other.value
    }

    pub fn __ne__(&self, other: &Self) -> bool {
        self.value != other.value
    }

    #[getter]
    pub fn column(&self) -> usize {
        self.get_column() as usize
    }

    #[setter]
    pub fn set_column(&mut self, column: usize) {
        let row: u8 = self.get_row();
        self.value = row * 11 + (column as u8);
    }

    #[getter]
    pub fn row(&self) -> usize {
        self.get_row() as usize
    }

    #[setter]
    pub fn set_row(&mut self, row: usize) {
        let column: u8 = self.get_column();
        self.value = (row as u8) * 11 + column;
    }

    #[pyo3(name = "to_int")]
    pub fn python_to_int(&self) -> usize {
        self.to_index() as usize
    }
}

pub fn index_to_row(index: u8) -> u8 {
    index / 11
}

pub fn index_to_column(index: u8) -> u8 {
    index % 11
}
