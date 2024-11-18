static int vble_unpack(VBLEContext *ctx, GetBitContext *gb)

{

    int i;

    static const uint8_t LUT[256] = {

        8,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,

        5,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,

        6,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,

        5,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,

        7,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,

        5,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,

        6,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,

        5,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,

    };



    /* Read all the lengths in first */

    for (i = 0; i < ctx->size; i++) {

        /* At most we need to read 9 bits total to get indices up to 8 */

        int val = show_bits(gb, 8);



        // read reverse unary

        if (val) {

            val = LUT[val];

            skip_bits(gb, val + 1);

            ctx->len[i] = val;

        } else {

            skip_bits(gb, 8);

            if (!get_bits1(gb))

                return -1;

            ctx->len[i] = 8;

        }

    }



    /* For any values that have length 0 */

    memset(ctx->val, 0, ctx->size);



    for (i = 0; i < ctx->size; i++) {

        /* Check we have enough bits left */

        if (get_bits_left(gb) < ctx->len[i])

            return -1;



        /* get_bits can't take a length of 0 */

        if (ctx->len[i])

            ctx->val[i] = (1 << ctx->len[i]) + get_bits(gb, ctx->len[i]) - 1;

    }



    return 0;

}
