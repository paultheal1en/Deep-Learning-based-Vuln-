static inline int get_egolomb(GetBitContext *gb)

{

    int v = 4;



    while (get_bits1(gb)) v++;



    return (1 << v) + get_bits(gb, v);

}
