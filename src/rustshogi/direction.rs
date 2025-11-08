use std::arch::x86_64::__m128i;

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum DirectionName {
    Up,
    UpLeft,
    Left,
    DownLeft,
    Down,
    DownRight,
    Right,
    UpRight,
    DirectionNameNumber = 8,
}

impl DirectionName {
    pub fn from_usize(n: usize) -> Self {
        match n {
            0 => Self::Up,
            1 => Self::UpLeft,
            2 => Self::Left,
            3 => Self::DownLeft,
            4 => Self::Down,
            5 => Self::DownRight,
            6 => Self::Right,
            7 => Self::UpRight,
            8 => Self::DirectionNameNumber,
            _ => Self::DirectionNameNumber,
        }
    }
}

#[derive(Clone)]
pub struct Direction {
    pub name: DirectionName,

    pub vertical_vector: i8,

    pub horizon_vector: i8,
}

impl Direction {
    pub fn new(name: DirectionName) -> Self {
        let (vertical_vector, horizon_vector) = match name {
            DirectionName::Up => (-1, 0),
            DirectionName::UpLeft => (-1, 1),
            DirectionName::Left => (0, 1),
            DirectionName::DownLeft => (1, 1),
            DirectionName::Down => (1, 0),
            DirectionName::DownRight => (1, -1),
            DirectionName::Right => (0, -1),
            DirectionName::UpRight => (-1, -1),
            DirectionName::DirectionNameNumber => (0, 0),
        };
        Self {
            name,
            vertical_vector,
            horizon_vector,
        }
    }

    pub fn reverse(&mut self) {
        self.vertical_vector *= -1;
        self.horizon_vector *= -1;
    }

    pub fn get_all_direction_vectors() -> &'static [(i8, i8); 8] {
        static DIRECTION_VECTORS: [(i8, i8); 8] = [
            (-1, 0),  // Up
            (-1, 1),  // UpLeft
            (0, 1),   // Left
            (1, 1),   // DownLeft
            (1, 0),   // Down
            (1, -1),  // DownRight
            (0, -1),  // Right
            (-1, -1), // UpRight
        ];
        &DIRECTION_VECTORS
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    pub unsafe fn get_all_direction_vectors_simd() -> (__m128i, __m128i) {
        use std::arch::x86_64::_mm_set_epi8;

        // Vectors are: Up, UpLeft, Left, DownLeft, Down, DownRight, Right, UpRight
        // vertical:   -1, -1, 0, 1, 1, 1, 0, -1
        // horizontal:  0,  1, 1, 1, 0,-1,-1, -1
        let vertical_vectors: __m128i = _mm_set_epi8(
            0, 0, 0, 0, 0, 0, 0, 0,  // padding
            -1, // UpRight
            0,  // Right
            1,  // DownRight
            1,  // Down
            1,  // DownLeft
            0,  // Left
            -1, // UpLeft
            -1, // Up
        );
        let horizontal_vectors: __m128i = _mm_set_epi8(
            0, 0, 0, 0, 0, 0, 0, 0,  // padding
            -1, // UpRight
            -1, // Right
            -1, // DownRight
            0,  // Down
            1,  // DownLeft
            1,  // Left
            1,  // UpLeft
            0,  // Up
        );

        (vertical_vectors, horizontal_vectors)
    }
}
