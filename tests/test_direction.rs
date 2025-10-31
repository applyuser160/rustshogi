use rustshogi::direction::{Direction, DirectionName};

#[test]
fn test_direction_from_usize() {
    let dn = DirectionName::from_usize(3);
    assert_eq!(dn, DirectionName::DownLeft);
}

#[test]
fn test_direction_new() {
    let direction = Direction::new(DirectionName::DownRight);
    assert_eq!(direction.name, DirectionName::DownRight);
    assert_eq!(direction.vertical_vector, 1);
    assert_eq!(direction.horizon_vector, -1);
}

#[test]
fn test_direction_reverse() {
    let mut direction = Direction::new(DirectionName::UpLeft);
    direction.reverse();
    assert_eq!(direction.name, DirectionName::UpLeft);
    assert_eq!(direction.vertical_vector, 1);
    assert_eq!(direction.horizon_vector, -1);
}
