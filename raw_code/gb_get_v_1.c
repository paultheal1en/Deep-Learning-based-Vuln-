static inline int64_t gb_get_v(GetBitContext *gb)

{

    int64_t v = 0;

    int bits = 0;

    while(get_bits1(gb) && bits < 64-7){

        v <<= 7;

        v |= get_bits(gb, 7);

        bits += 7;

    }

    v <<= 7;

    v |= get_bits(gb, 7);



    return v;

}
