static int rv40_decode_intra_types(RV34DecContext *r, GetBitContext *gb, int8_t *dst)

{

    MpegEncContext *s = &r->s;

    int i, j, k, v;

    int A, B, C;

    int pattern;

    int8_t *ptr;



    for(i = 0; i < 4; i++, dst += r->intra_types_stride){

        if(!i && s->first_slice_line){

            pattern = get_vlc2(gb, aic_top_vlc.table, AIC_TOP_BITS, 1);

            dst[0] = (pattern >> 2) & 2;

            dst[1] = (pattern >> 1) & 2;

            dst[2] =  pattern       & 2;

            dst[3] = (pattern << 1) & 2;

            continue;

        }

        ptr = dst;

        for(j = 0; j < 4; j++){

            /* Coefficients are read using VLC chosen by the prediction pattern

             * The first one (used for retrieving a pair of coefficients) is

             * constructed from the top, top right and left coefficients

             * The second one (used for retrieving only one coefficient) is

             * top + 10 * left.

             */

            A = ptr[-r->intra_types_stride + 1]; // it won't be used for the last coefficient in a row

            B = ptr[-r->intra_types_stride];

            C = ptr[-1];

            pattern = A + (B << 4) + (C << 8);

            for(k = 0; k < MODE2_PATTERNS_NUM; k++)

                if(pattern == rv40_aic_table_index[k])

                    break;

            if(j < 3 && k < MODE2_PATTERNS_NUM){ //pattern is found, decoding 2 coefficients

                v = get_vlc2(gb, aic_mode2_vlc[k].table, AIC_MODE2_BITS, 2);

                *ptr++ = v/9;

                *ptr++ = v%9;

                j++;

            }else{

                if(B != -1 && C != -1)

                    v = get_vlc2(gb, aic_mode1_vlc[B + C*10].table, AIC_MODE1_BITS, 1);

                else{ // tricky decoding

                    v = 0;

                    switch(C){

                    case -1: // code 0 -> 1, 1 -> 0

                        if(B < 2)

                            v = get_bits1(gb) ^ 1;

                        break;

                    case  0:

                    case  2: // code 0 -> 2, 1 -> 0

                        v = (get_bits1(gb) ^ 1) << 1;

                        break;

                    }

                }

                *ptr++ = v;

            }

        }

    }

    return 0;

}
