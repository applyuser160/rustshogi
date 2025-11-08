use pyo3::prelude::*;

#[pyclass(eq, eq_int)]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
#[repr(i8)]
pub enum ColorType {
    None = -1,
    Black = 0,
    White,
    ColorNumber = 2,
}

impl ColorType {
    pub fn from_u8(n: u8) -> Self {
        match n {
            0 => Self::Black,
            1 => Self::White,
            2 => Self::ColorNumber,
            _ => Self::None,
        }
    }

    pub fn to_bool(&self) -> bool {
        self == &ColorType::White
    }

    pub fn get_name(&self) -> &'static str {
        match self {
            ColorType::None => "None",
            ColorType::Black => "Black",
            ColorType::White => "White",
            ColorType::ColorNumber => "ColorNumber",
        }
    }
}

#[pymethods]
impl ColorType {
    #[new]
    pub fn new(value: u8) -> Self {
        Self::from_u8(value)
    }

    pub fn __repr__(&self) -> String {
        format!("<ColorType.{}: {}>", self.get_name(), *self as u8)
    }

    pub fn __str__(&self) -> String {
        format!("ColorType.{}", self.get_name())
    }

    #[getter]
    pub fn name(&self) -> String {
        self.get_name().to_string()
    }

    #[getter]
    pub fn value(&self) -> u8 {
        *self as u8
    }
}

pub fn get_reverse_color(color_type: ColorType) -> ColorType {
    match color_type {
        ColorType::Black => ColorType::White,
        ColorType::White => ColorType::Black,
        _ => ColorType::None,
    }
}

pub fn convert_string(color_type: ColorType) -> char {
    match color_type {
        ColorType::Black => 'b',
        ColorType::White => 'w',
        _ => 'd',
    }
}

pub fn convert_from_string(character: char) -> ColorType {
    match character {
        'b' => ColorType::Black,
        'w' => ColorType::White,
        _ => ColorType::None,
    }
}

impl std::fmt::Display for ColorType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", convert_string(*self))
    }
}
