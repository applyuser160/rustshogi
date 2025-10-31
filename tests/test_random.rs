use rustshogi::random::Random;

#[test]
fn test_random_generate_one1() {
    let mut rand = Random::init();
    let n1 = rand.generate_one();
    assert!(n1 <= 9);
}

#[test]
fn test_random_generate_one2() {
    let mut rand = Random::new(5, 16);
    let n1 = rand.generate_one();
    assert!(n1 >= 5);
    assert!(n1 <= 16);
}
