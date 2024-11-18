static inline int decode_scalar(GetBitContext *gb, int k, int limit, int readsamplesize){

    int x = get_unary_0_9(gb);



    if (x > 8) { /* RICE THRESHOLD */

        /* use alternative encoding */

        x = get_bits(gb, readsamplesize);

    } else {

        if (k >= limit)

            k = limit;



        if (k != 1) {

            int extrabits = show_bits(gb, k);



            /* multiply x by 2^k - 1, as part of their strange algorithm */

            x = (x << k) - x;



            if (extrabits > 1) {

                x += extrabits - 1;

                skip_bits(gb, k);

            } else

                skip_bits(gb, k - 1);

        }

    }

    return x;

}
