static inline int ape_decode_value_3860(APEContext *ctx, GetBitContext *gb,

                                        APERice *rice)

{

    unsigned int x, overflow;



    overflow = get_unary(gb, 1, get_bits_left(gb));



    if (ctx->fileversion > 3880) {

        while (overflow >= 16) {

            overflow -= 16;

            rice->k  += 4;

        }

    }



    if (!rice->k)

        x = overflow;

    else

        x = (overflow << rice->k) + get_bits(gb, rice->k);



    rice->ksum += x - (rice->ksum + 8 >> 4);

    if (rice->ksum < (rice->k ? 1 << (rice->k + 4) : 0))

        rice->k--;

    else if (rice->ksum >= (1 << (rice->k + 5)) && rice->k < 24)

        rice->k++;



    /* Convert to signed */

    if (x & 1)

        return (x >> 1) + 1;

    else

        return -(x >> 1);

}
