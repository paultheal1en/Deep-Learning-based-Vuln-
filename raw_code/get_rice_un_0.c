static int get_rice_un(GetBitContext *gb, int k)

{

    unsigned int v = get_unary(gb, 1, 128);

    return (v << k) | get_bits_long(gb, k);

}
