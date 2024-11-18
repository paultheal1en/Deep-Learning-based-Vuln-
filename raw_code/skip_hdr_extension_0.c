static inline void skip_hdr_extension(GetBitContext *gb)

{

    int i, len;



    do {

        len = get_bits(gb, 8);

        for (i = 0; i < len; i++) skip_bits(gb, 8);

    } while(len);

}
