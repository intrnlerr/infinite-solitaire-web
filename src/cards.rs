#[derive(Debug, Clone, Copy)]
/// a BitCard has the following layout
///
/// ```
/// extra |-pip-| |-suit-|
///   7   6     2 1      0
/// ```
pub struct BitCard(u8);

// also a bitcard is any number from 0 to 51 (inclusive)

impl BitCard {
    fn color(self) -> u8 {
        ((self.0 >> 1) ^ self.0) & 1
    }

    /// returns the "number" or "pips" or whatever the fuck you call it
    ///
    /// - 0 -> A
    /// - 1 -> 2
    /// - 2 -> 3
    /// -   ...
    /// - 12 -> K
    pub fn number(self) -> u8 {
        self.0 >> 2
    }

    pub fn is_ace(self) -> bool {
        self.number() == 0
    }

    fn suit_raw(self) -> u8 {
        self.0 & 0b11
    }

    // pub fn suit(self) -> Suit {
    //     match self.0 & 0b11 {
    //         0b00 => Suit::Diamond,
    //         0b01 => Suit::Heart,
    //         0b10 => Suit::Club,
    //         0b11 => Suit::Spade,
    //         _ => unreachable!(),
    //     }
    // }
    // 00 -> clubs
    // 01 -> diamonds
    // 10 -> hearts
    // 11 -> spades

    fn is_king(self) -> bool {
        self.number() == 12
    }

    pub fn is_next_card(self, other: Self) -> bool {
        other.number() + 1 == self.number()
    }

    pub fn same_suit(self, other: Self) -> bool {
        self.suit_raw() == other.suit_raw()
    }

    fn can_stack_on(self, under: Self) -> bool {
        self.color() != under.color() && under.number() == self.number() + 1
    }

    // note to future: if i ever want to replace this with xoroshiro or something,
    // https://stackoverflow.com/questions/10984974/why-do-people-say-there-is-modulo-bias-when-using-a-random-number-generator
    // the best is 62 bits
    fn random(rng: &mut oorandom::Rand32) -> Self {
        Self(rng.rand_range(0..52) as u8)
    }

    pub fn as_u8(self) -> u8 {
        self.0
    }
}

pub struct CardStack {
    // since we only have at most 13 cards, a byte is small enough to store the size
    len: u8,
    // it is not possible to stack more than K -> A, so only 13 are needed
    cards: [BitCard; 13],
}

impl CardStack {
    pub fn empty() -> Self {
        Self {
            len: 0,
            cards: [BitCard(0); 13],
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn can_stack(&self, top_card: BitCard) -> bool {
        if self.is_empty() {
            top_card.is_king()
        } else {
            top_card.can_stack_on(self.cards[(self.len - 1) as usize])
        }
    }

    fn one_random(rng: &mut oorandom::Rand32) -> Self {
        let mut cards = [BitCard(0); 13];
        cards[0] = BitCard::random(rng);
        Self { len: 1, cards }
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = BitCard> + '_ {
        self.cards[..(self.len as usize)].iter().cloned()
    }

    pub fn len(&self) -> u8 {
        self.len
    }

    pub(crate) fn take_from(&mut self, other: &mut Self, visible_idx: usize) {
        debug_assert!(other.len as usize > visible_idx);
        let idx = visible_idx as u8;
        let taken = &other.cards[visible_idx..other.len as usize];
        self.len = taken.len() as u8;
        self.cards[..taken.len()].copy_from_slice(taken);
        other.len = idx;
    }

    pub(crate) fn append(&mut self, from: &mut Self) {
        debug_assert!(self.len + from.len <= 13);
        self.cards[self.len as usize..(self.len + from.len) as usize]
            .copy_from_slice(&from.cards[..from.len as usize]);
        self.len += from.len;
        from.len = 0;
    }

    pub fn top(&self) -> BitCard {
        self.cards[0]
    }

    pub fn clear(&mut self) {
        self.len = 0
    }
}

pub struct Column {
    visible: CardStack,
    pub under: u32,
}

impl Column {
    pub fn new(under: u32, rng: &mut oorandom::Rand32) -> Self {
        Self {
            visible: CardStack::one_random(rng),
            under,
        }
    }

    pub fn is_visible_empty(&self) -> bool {
        self.visible.len == 0
    }

    pub fn visible(&self) -> &CardStack {
        &self.visible
    }

    pub(crate) fn maybe_reveal_card(&mut self, rng: &mut oorandom::Rand32) {
        if !(self.under > 0 && self.is_visible_empty()) {
            return;
        }
        self.visible.cards[0] = BitCard::random(rng);
        self.visible.len = 1;
        self.under -= 1;
    }

    pub fn append(&mut self, from: &mut CardStack) {
        self.visible.append(from)
    }

    pub(crate) fn visible_mut(&mut self) -> &mut CardStack {
        &mut self.visible
    }
}
