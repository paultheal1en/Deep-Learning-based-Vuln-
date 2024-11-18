static int decode_cabac_residual( H264Context *h, DCTELEM *block, int cat, int n, const uint8_t *scantable, const uint32_t *qmul, int max_coeff) {

    const int mb_xy  = h->s.mb_x + h->s.mb_y*h->s.mb_stride;

    static const int significant_coeff_flag_offset[2][6] = {

      { 105+0, 105+15, 105+29, 105+44, 105+47, 402 },

      { 277+0, 277+15, 277+29, 277+44, 277+47, 436 }

    };

    static const int last_coeff_flag_offset[2][6] = {

      { 166+0, 166+15, 166+29, 166+44, 166+47, 417 },

      { 338+0, 338+15, 338+29, 338+44, 338+47, 451 }

    };

    static const int coeff_abs_level_m1_offset[6] = {

        227+0, 227+10, 227+20, 227+30, 227+39, 426

    };

    static const uint8_t significant_coeff_flag_offset_8x8[2][63] = {

      { 0, 1, 2, 3, 4, 5, 5, 4, 4, 3, 3, 4, 4, 4, 5, 5,

        4, 4, 4, 4, 3, 3, 6, 7, 7, 7, 8, 9,10, 9, 8, 7,

        7, 6,11,12,13,11, 6, 7, 8, 9,14,10, 9, 8, 6,11,

       12,13,11, 6, 9,14,10, 9,11,12,13,11,14,10,12 },

      { 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 7, 7, 8, 4, 5,

        6, 9,10,10, 8,11,12,11, 9, 9,10,10, 8,11,12,11,

        9, 9,10,10, 8,11,12,11, 9, 9,10,10, 8,13,13, 9,

        9,10,10, 8,13,13, 9, 9,10,10,14,14,14,14,14 }

    };

    static const uint8_t last_coeff_flag_offset_8x8[63] = {

        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,

        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,

        3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4,

        5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8

    };



    int index[64];



    int i, last;

    int coeff_count = 0;



    int abslevel1 = 1;

    int abslevelgt1 = 0;



    uint8_t *significant_coeff_ctx_base;

    uint8_t *last_coeff_ctx_base;

    uint8_t *abs_level_m1_ctx_base;



    /* cat: 0-> DC 16x16  n = 0

     *      1-> AC 16x16  n = luma4x4idx

     *      2-> Luma4x4   n = luma4x4idx

     *      3-> DC Chroma n = iCbCr

     *      4-> AC Chroma n = 4 * iCbCr + chroma4x4idx

     *      5-> Luma8x8   n = 4 * luma8x8idx

     */



    /* read coded block flag */

    if( cat != 5 ) {

        if( get_cabac( &h->cabac, &h->cabac_state[85 + get_cabac_cbf_ctx( h, cat, n ) ] ) == 0 ) {

            if( cat == 1 || cat == 2 )

                h->non_zero_count_cache[scan8[n]] = 0;

            else if( cat == 4 )

                h->non_zero_count_cache[scan8[16+n]] = 0;



            return 0;

        }

    }



    significant_coeff_ctx_base = h->cabac_state

        + significant_coeff_flag_offset[MB_FIELD][cat];

    last_coeff_ctx_base = h->cabac_state

        + last_coeff_flag_offset[MB_FIELD][cat];

    abs_level_m1_ctx_base = h->cabac_state

        + coeff_abs_level_m1_offset[cat];



    if( cat == 5 ) {

#define DECODE_SIGNIFICANCE( coefs, sig_off, last_off ) \

        for(last= 0; last < coefs; last++) { \

            uint8_t *sig_ctx = significant_coeff_ctx_base + sig_off; \

            if( get_cabac( &h->cabac, sig_ctx )) { \

                uint8_t *last_ctx = last_coeff_ctx_base + last_off; \

                index[coeff_count++] = last; \

                if( get_cabac( &h->cabac, last_ctx ) ) { \

                    last= max_coeff; \

                    break; \

                } \

            } \

        }

        const uint8_t *sig_off = significant_coeff_flag_offset_8x8[MB_FIELD];

        DECODE_SIGNIFICANCE( 63, sig_off[last], last_coeff_flag_offset_8x8[last] );

    } else {

        DECODE_SIGNIFICANCE( max_coeff - 1, last, last );

    }

    if( last == max_coeff -1 ) {

        index[coeff_count++] = last;

    }

    assert(coeff_count > 0);



    if( cat == 0 )

        h->cbp_table[mb_xy] |= 0x100;

    else if( cat == 1 || cat == 2 )

        h->non_zero_count_cache[scan8[n]] = coeff_count;

    else if( cat == 3 )

        h->cbp_table[mb_xy] |= 0x40 << n;

    else if( cat == 4 )

        h->non_zero_count_cache[scan8[16+n]] = coeff_count;

    else {

        assert( cat == 5 );

        fill_rectangle(&h->non_zero_count_cache[scan8[n]], 2, 2, 8, coeff_count, 1);

    }



    for( i = coeff_count - 1; i >= 0; i-- ) {

        uint8_t *ctx = (abslevelgt1 != 0 ? 0 : FFMIN( 4, abslevel1 )) + abs_level_m1_ctx_base;

        int j= scantable[index[i]];



        if( get_cabac( &h->cabac, ctx ) == 0 ) {

            if( !qmul ) {

                if( get_cabac_bypass( &h->cabac ) ) block[j] = -1;

                else                                block[j] =  1;

            }else{

                if( get_cabac_bypass( &h->cabac ) ) block[j] = (-qmul[j] + 32) >> 6;

                else                                block[j] = ( qmul[j] + 32) >> 6;

            }



            abslevel1++;

        } else {

            int coeff_abs = 2;

            ctx = 5 + FFMIN( 4, abslevelgt1 ) + abs_level_m1_ctx_base;

            while( coeff_abs < 15 && get_cabac( &h->cabac, ctx ) ) {

                coeff_abs++;

            }



            if( coeff_abs >= 15 ) {

                int j = 0;

                while( get_cabac_bypass( &h->cabac ) ) {

                    j++;

                }



                coeff_abs=1;

                while( j-- ) {

                    coeff_abs += coeff_abs + get_cabac_bypass( &h->cabac );

                }

                coeff_abs+= 14;

            }



            if( !qmul ) {

                if( get_cabac_bypass( &h->cabac ) ) block[j] = -coeff_abs;

                else                                block[j] =  coeff_abs;

            }else{

                if( get_cabac_bypass( &h->cabac ) ) block[j] = (-coeff_abs * qmul[j] + 32) >> 6;

                else                                block[j] = ( coeff_abs * qmul[j] + 32) >> 6;

            }



            abslevelgt1++;

        }

    }

    return 0;

}
