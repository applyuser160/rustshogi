use crate::direction::DirectionName;
use crate::piece::MoveType;

pub struct MovePattern {
    pub move_type: MoveType,
    pub direction: DirectionName,
}

pub const KING_MOVE_PATTERNS: &[MovePattern] = &[
    MovePattern {
        move_type: MoveType::Short,
        direction: DirectionName::Up,
    },
    MovePattern {
        move_type: MoveType::Short,
        direction: DirectionName::UpLeft,
    },
    MovePattern {
        move_type: MoveType::Short,
        direction: DirectionName::Left,
    },
    MovePattern {
        move_type: MoveType::Short,
        direction: DirectionName::DownLeft,
    },
    MovePattern {
        move_type: MoveType::Short,
        direction: DirectionName::Down,
    },
    MovePattern {
        move_type: MoveType::Short,
        direction: DirectionName::DownRight,
    },
    MovePattern {
        move_type: MoveType::Short,
        direction: DirectionName::Right,
    },
    MovePattern {
        move_type: MoveType::Short,
        direction: DirectionName::UpRight,
    },
];

pub const GOLD_MOVE_PATTERNS: &[MovePattern] = &[
    MovePattern {
        move_type: MoveType::Short,
        direction: DirectionName::Up,
    },
    MovePattern {
        move_type: MoveType::Short,
        direction: DirectionName::UpLeft,
    },
    MovePattern {
        move_type: MoveType::Short,
        direction: DirectionName::Left,
    },
    MovePattern {
        move_type: MoveType::Short,
        direction: DirectionName::Down,
    },
    MovePattern {
        move_type: MoveType::Short,
        direction: DirectionName::Right,
    },
    MovePattern {
        move_type: MoveType::Short,
        direction: DirectionName::UpRight,
    },
];

pub const ROOK_MOVE_PATTERNS: &[MovePattern] = &[
    MovePattern {
        move_type: MoveType::Long,
        direction: DirectionName::Up,
    },
    MovePattern {
        move_type: MoveType::Long,
        direction: DirectionName::Left,
    },
    MovePattern {
        move_type: MoveType::Long,
        direction: DirectionName::Down,
    },
    MovePattern {
        move_type: MoveType::Long,
        direction: DirectionName::Right,
    },
];

pub const BISHOP_MOVE_PATTERNS: &[MovePattern] = &[
    MovePattern {
        move_type: MoveType::Long,
        direction: DirectionName::UpLeft,
    },
    MovePattern {
        move_type: MoveType::Long,
        direction: DirectionName::DownLeft,
    },
    MovePattern {
        move_type: MoveType::Long,
        direction: DirectionName::DownRight,
    },
    MovePattern {
        move_type: MoveType::Long,
        direction: DirectionName::UpRight,
    },
];

pub const SILVER_MOVE_PATTERNS: &[MovePattern] = &[
    MovePattern {
        move_type: MoveType::Short,
        direction: DirectionName::Up,
    },
    MovePattern {
        move_type: MoveType::Short,
        direction: DirectionName::UpLeft,
    },
    MovePattern {
        move_type: MoveType::Short,
        direction: DirectionName::DownLeft,
    },
    MovePattern {
        move_type: MoveType::Short,
        direction: DirectionName::DownRight,
    },
    MovePattern {
        move_type: MoveType::Short,
        direction: DirectionName::UpRight,
    },
];

pub const KNIGHT_MOVE_PATTERNS: &[MovePattern] = &[
    MovePattern {
        move_type: MoveType::Hop,
        direction: DirectionName::UpLeft,
    },
    MovePattern {
        move_type: MoveType::Hop,
        direction: DirectionName::UpRight,
    },
];

pub const LANCE_MOVE_PATTERNS: &[MovePattern] = &[MovePattern {
    move_type: MoveType::Long,
    direction: DirectionName::Up,
}];

pub const PAWN_MOVE_PATTERNS: &[MovePattern] = &[MovePattern {
    move_type: MoveType::Short,
    direction: DirectionName::Up,
}];

pub const DRAGON_MOVE_PATTERNS: &[MovePattern] = &[
    MovePattern {
        move_type: MoveType::Long,
        direction: DirectionName::Up,
    },
    MovePattern {
        move_type: MoveType::Long,
        direction: DirectionName::Left,
    },
    MovePattern {
        move_type: MoveType::Long,
        direction: DirectionName::Down,
    },
    MovePattern {
        move_type: MoveType::Long,
        direction: DirectionName::Right,
    },
    MovePattern {
        move_type: MoveType::Short,
        direction: DirectionName::UpLeft,
    },
    MovePattern {
        move_type: MoveType::Short,
        direction: DirectionName::DownLeft,
    },
    MovePattern {
        move_type: MoveType::Short,
        direction: DirectionName::DownRight,
    },
    MovePattern {
        move_type: MoveType::Short,
        direction: DirectionName::UpRight,
    },
];

pub const HORSE_MOVE_PATTERNS: &[MovePattern] = &[
    MovePattern {
        move_type: MoveType::Long,
        direction: DirectionName::UpLeft,
    },
    MovePattern {
        move_type: MoveType::Long,
        direction: DirectionName::DownLeft,
    },
    MovePattern {
        move_type: MoveType::Long,
        direction: DirectionName::DownRight,
    },
    MovePattern {
        move_type: MoveType::Long,
        direction: DirectionName::UpRight,
    },
    MovePattern {
        move_type: MoveType::Short,
        direction: DirectionName::Up,
    },
    MovePattern {
        move_type: MoveType::Short,
        direction: DirectionName::Left,
    },
    MovePattern {
        move_type: MoveType::Short,
        direction: DirectionName::Down,
    },
    MovePattern {
        move_type: MoveType::Short,
        direction: DirectionName::Right,
    },
];

// Promoted pieces have the same move patterns as Gold
pub const PRO_SILVER_MOVE_PATTERNS: &[MovePattern] = GOLD_MOVE_PATTERNS;
pub const PRO_KNIGHT_MOVE_PATTERNS: &[MovePattern] = GOLD_MOVE_PATTERNS;
pub const PRO_LANCE_MOVE_PATTERNS: &[MovePattern] = GOLD_MOVE_PATTERNS;
pub const PRO_PAWN_MOVE_PATTERNS: &[MovePattern] = GOLD_MOVE_PATTERNS;
