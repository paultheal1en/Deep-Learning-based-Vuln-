static int can_safely_read(GetBitContext* gb, uint64_t bits) {

    return get_bits_left(gb) >= bits;

}
