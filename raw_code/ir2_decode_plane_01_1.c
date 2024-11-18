static void ir2_decode_plane(Ir2Context *ctx, int width, int height, uint8_t *dst, int stride,

                             const uint8_t *table)

{

    int i;

    int j;

    int out = 0;

    int c;

    int t;

    

    /* first line contain absolute values, other lines contain deltas */

    while (out < width){

        c = ir2_get_code(&ctx->gb);

        if(c > 0x80) { /* we have a run */

            c -= 0x80;

            for (i = 0; i < c * 2; i++)

                dst[out++] = 0x80;

        } else { /* copy two values from table */

            dst[out++] = table[c * 2];

            dst[out++] = table[(c * 2) + 1];

        }

    }

    dst += stride;

    

    for (j = 1; j < height; j++){

        out = 0;

        while (out < width){

            c = ir2_get_code(&ctx->gb);

            if(c > 0x80) { /* we have a skip */

                c -= 0x80;

                for (i = 0; i < c * 2; i++) {

                    dst[out] = dst[out - stride];

                    out++;

                }

            } else { /* add two deltas from table */

                t = dst[out - stride] + (table[c * 2] - 128);

                CLAMP_TO_BYTE(t);

                dst[out] = t;

                out++;

                t = dst[out - stride] + (table[(c * 2) + 1] - 128);

                CLAMP_TO_BYTE(t);

                dst[out] = t;

                out++;

            }

        }

        dst += stride;

    }

}
