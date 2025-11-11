use crate::address::Address;
use crate::piece::Piece;

use super::address;
use super::piece;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Debug)]
pub struct Move {
    pub value: u16,
}

impl Move {
    fn is_drop(csa: &str) -> bool {
        csa.chars().nth(1) == Some('*')
    }

    fn is_promote(csa: &str) -> bool {
        if csa.len() > 4 {
            csa.chars().nth(4) == Some('+')
        } else {
            false
        }
    }

    fn base_constructor(&mut self, from: u16, to: u16, pro: u16, drop: u16) {
        // Bit layout:
        // bit 15: drop (1 bit)
        // bit 14: promote (1 bit)
        // bits 13-7: to (7 bits)
        // bits 6-0: from/piece (7 bits)
        self.value = (drop << 15) | (pro << 14) | (to << 7) | from;
    }

    fn standard_constructor(
        &mut self,
        from: address::Address,
        to: address::Address,
        promote: bool,
    ) {
        self.base_constructor(
            from.to_index() as u16,
            to.to_index() as u16,
            promote as u16,
            0,
        );
    }

    fn drop_constructor(&mut self, piece: piece::Piece, to: address::Address) {
        self.base_constructor(piece.to_u8() as u16, to.to_index() as u16, 0, 1)
    }

    pub fn new() -> Self {
        Self { value: 0 }
    }

    pub fn from_standard(from: address::Address, to: address::Address, promote: bool) -> Self {
        let mut res: Move = Self::new();
        res.standard_constructor(from, to, promote);
        res
    }

    pub fn from_drop(piece: piece::Piece, to: address::Address) -> Self {
        let mut res: Move = Self::new();
        res.drop_constructor(piece, to);
        res
    }

    pub fn from_csa(csa: &str) -> Self {
        let mut res: Move = Self::new();
        let to: Address = address::Address::from_string(&csa[2..]);
        if Self::is_drop(csa) {
            let piece: Piece = piece::Piece::from_char(csa.chars().nth(0).unwrap());
            res.drop_constructor(piece, to);
        } else {
            let from: Address = address::Address::from_string(csa);
            res.standard_constructor(from, to, Self::is_promote(csa));
        }
        res
    }

    pub fn get_is_drop(&self) -> bool {
        (self.value & (1 << 15)) != 0
    }

    pub fn get_is_promote(&self) -> bool {
        (self.value & (1 << 14)) != 0
    }

    pub fn get_from(&self) -> address::Address {
        // bits 6-0: from (7 bits)
        let v: u8 = (self.value & 0x7F) as u8;
        address::Address::from_number(v)
    }

    pub fn get_to(&self) -> address::Address {
        // bits 13-7: to (7 bits)
        let v: u8 = ((self.value >> 7) & 0x7F) as u8;
        address::Address::from_number(v)
    }

    pub fn get_piece(&self) -> piece::Piece {
        // For a drop move, the piece information is stored in the lower 7 bits (bits 0-6)
        let v: u8 = (self.value & 0x7F) as u8;
        piece::Piece::from_u8(v)
    }

    pub fn to_string(&self) -> String {
        let mut first: String = String::with_capacity(2);
        if self.get_is_drop() {
            let piece: Piece = self.get_piece();
            first.push_str(&piece.to_string());
            first.push('*');
        } else {
            let from: Address = self.get_from();
            first.push_str(&from.to_string());
        }
        let to: Address = self.get_to();
        let is_pro: bool = self.get_is_promote();
        let mut res: String = String::with_capacity(5);
        res.push_str(&first);
        res.push_str(&to.to_string());
        if is_pro {
            res.push('+');
        }
        res
    }
}

#[pymethods]
impl Move {
    #[new]
    #[pyo3(signature = (csa = None, from_address = None, to_address = None, promote = false, piece = None))]
    pub fn new_for_python(
        csa: Option<String>,
        from_address: Option<Address>,
        to_address: Option<Address>,
        promote: bool,
        piece: Option<Piece>,
    ) -> PyResult<Self> {
        match (csa, from_address, to_address, piece) {
            (Some(csa), _, _, _) => Ok(Self::from_csa(csa.as_str())),
            (None, Some(from), Some(to), _) => Ok(Self::from_standard(from, to, promote)),
            (None, Some(_), None, _) => Err(PyValueError::new_err(
                "to_address must be provided when from_address is set",
            )),
            (None, None, Some(to), Some(piece)) => Ok(Self::from_drop(piece, to)),
            (None, None, None, Some(_)) => Err(PyValueError::new_err(
                "to_address must be provided when piece is set",
            )),
            (None, None, Some(_), None) => Err(PyValueError::new_err(
                "to_address requires either from_address or piece",
            )),
            (None, None, None, None) => Ok(Self::new()),
        }
    }

    pub fn __repr__(&self) -> String {
        format!("Move(csa={})", self.to_string())
    }

    pub fn __str__(&self) -> String {
        format!("Move(csa={})", self.to_string())
    }

    pub fn __eq__(&self, other: &Self) -> bool {
        self.value == other.value
    }

    pub fn __ne__(&self, other: &Self) -> bool {
        self.value != other.value
    }

    #[pyo3(name = "is_drop")]
    pub fn python_is_drop(&self) -> bool {
        self.get_is_drop()
    }

    #[pyo3(name = "is_promote")]
    pub fn python_is_promote(&self) -> bool {
        self.get_is_promote()
    }

    #[pyo3(name = "get_from")]
    pub fn python_get_from(&self) -> address::Address {
        self.get_from()
    }

    #[pyo3(name = "get_to")]
    pub fn python_get_to(&self) -> address::Address {
        self.get_to()
    }

    #[pyo3(name = "get_piece")]
    pub fn python_get_piece(&self) -> piece::Piece {
        self.get_piece()
    }
}

impl std::fmt::Display for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}
