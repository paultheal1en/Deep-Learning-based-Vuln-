int ff_h264_decode_mb_cabac(const H264Context *h, H264SliceContext *sl)

{

    int mb_xy;

    int mb_type, partition_count, cbp = 0;

    int dct8x8_allowed= h->pps.transform_8x8_mode;

    int decode_chroma = h->sps.chroma_format_idc == 1 || h->sps.chroma_format_idc == 2;

    const int pixel_shift = h->pixel_shift;



    mb_xy = sl->mb_xy = sl->mb_x + sl->mb_y*h->mb_stride;



    ff_tlog(h->avctx, "pic:%d mb:%d/%d\n", h->frame_num, sl->mb_x, sl->mb_y);

    if (sl->slice_type_nos != AV_PICTURE_TYPE_I) {

        int skip;

        /* a skipped mb needs the aff flag from the following mb */

        if (FRAME_MBAFF(h) && (sl->mb_y & 1) == 1 && sl->prev_mb_skipped)

            skip = sl->next_mb_skipped;

        else

            skip = decode_cabac_mb_skip(h, sl, sl->mb_x, sl->mb_y );

        /* read skip flags */

        if( skip ) {

            if (FRAME_MBAFF(h) && (sl->mb_y & 1) == 0) {

                h->cur_pic.mb_type[mb_xy] = MB_TYPE_SKIP;

                sl->next_mb_skipped = decode_cabac_mb_skip(h, sl, sl->mb_x, sl->mb_y+1 );

                if(!sl->next_mb_skipped)

                    sl->mb_mbaff = sl->mb_field_decoding_flag = decode_cabac_field_decoding_flag(h, sl);

            }



            decode_mb_skip(h, sl);



            h->cbp_table[mb_xy] = 0;

            h->chroma_pred_mode_table[mb_xy] = 0;

            sl->last_qscale_diff = 0;



            return 0;



        }

    }

    if (FRAME_MBAFF(h)) {

        if ((sl->mb_y & 1) == 0)

            sl->mb_mbaff =

            sl->mb_field_decoding_flag = decode_cabac_field_decoding_flag(h, sl);

    }



    sl->prev_mb_skipped = 0;



    fill_decode_neighbors(h, sl, -(MB_FIELD(sl)));



    if (sl->slice_type_nos == AV_PICTURE_TYPE_B) {

        int ctx = 0;

        av_assert2(sl->slice_type_nos == AV_PICTURE_TYPE_B);



        if (!IS_DIRECT(sl->left_type[LTOP] - 1))

            ctx++;

        if (!IS_DIRECT(sl->top_type - 1))

            ctx++;



        if( !get_cabac_noinline( &sl->cabac, &sl->cabac_state[27+ctx] ) ){

            mb_type= 0; /* B_Direct_16x16 */

        }else if( !get_cabac_noinline( &sl->cabac, &sl->cabac_state[27+3] ) ) {

            mb_type= 1 + get_cabac_noinline( &sl->cabac, &sl->cabac_state[27+5] ); /* B_L[01]_16x16 */

        }else{

            int bits;

            bits = get_cabac_noinline( &sl->cabac, &sl->cabac_state[27+4] ) << 3;

            bits+= get_cabac_noinline( &sl->cabac, &sl->cabac_state[27+5] ) << 2;

            bits+= get_cabac_noinline( &sl->cabac, &sl->cabac_state[27+5] ) << 1;

            bits+= get_cabac_noinline( &sl->cabac, &sl->cabac_state[27+5] );

            if( bits < 8 ){

                mb_type= bits + 3; /* B_Bi_16x16 through B_L1_L0_16x8 */

            }else if( bits == 13 ){

                mb_type = decode_cabac_intra_mb_type(sl, 32, 0);

                goto decode_intra_mb;

            }else if( bits == 14 ){

                mb_type= 11; /* B_L1_L0_8x16 */

            }else if( bits == 15 ){

                mb_type= 22; /* B_8x8 */

            }else{

                bits= ( bits<<1 ) + get_cabac_noinline( &sl->cabac, &sl->cabac_state[27+5] );

                mb_type= bits - 4; /* B_L0_Bi_* through B_Bi_Bi_* */

            }

        }

            partition_count= b_mb_type_info[mb_type].partition_count;

            mb_type=         b_mb_type_info[mb_type].type;

    } else if (sl->slice_type_nos == AV_PICTURE_TYPE_P) {

        if( get_cabac_noinline( &sl->cabac, &sl->cabac_state[14] ) == 0 ) {

            /* P-type */

            if( get_cabac_noinline( &sl->cabac, &sl->cabac_state[15] ) == 0 ) {

                /* P_L0_D16x16, P_8x8 */

                mb_type= 3 * get_cabac_noinline( &sl->cabac, &sl->cabac_state[16] );

            } else {

                /* P_L0_D8x16, P_L0_D16x8 */

                mb_type= 2 - get_cabac_noinline( &sl->cabac, &sl->cabac_state[17] );

            }

            partition_count= p_mb_type_info[mb_type].partition_count;

            mb_type=         p_mb_type_info[mb_type].type;

        } else {

            mb_type = decode_cabac_intra_mb_type(sl, 17, 0);

            goto decode_intra_mb;

        }

    } else {

        mb_type = decode_cabac_intra_mb_type(sl, 3, 1);

        if (sl->slice_type == AV_PICTURE_TYPE_SI && mb_type)

            mb_type--;

        av_assert2(sl->slice_type_nos == AV_PICTURE_TYPE_I);

decode_intra_mb:

        partition_count = 0;

        cbp= i_mb_type_info[mb_type].cbp;

        sl->intra16x16_pred_mode = i_mb_type_info[mb_type].pred_mode;

        mb_type= i_mb_type_info[mb_type].type;

    }

    if (MB_FIELD(sl))

        mb_type |= MB_TYPE_INTERLACED;



    h->slice_table[mb_xy] = sl->slice_num;



    if(IS_INTRA_PCM(mb_type)) {

        const int mb_size = ff_h264_mb_sizes[h->sps.chroma_format_idc] *

                            h->sps.bit_depth_luma >> 3;

        const uint8_t *ptr;



        // We assume these blocks are very rare so we do not optimize it.

        // FIXME The two following lines get the bitstream position in the cabac

        // decode, I think it should be done by a function in cabac.h (or cabac.c).

        ptr= sl->cabac.bytestream;

        if(sl->cabac.low&0x1) ptr--;

        if(CABAC_BITS==16){

            if(sl->cabac.low&0x1FF) ptr--;

        }



        // The pixels are stored in the same order as levels in h->mb array.

        if ((int) (sl->cabac.bytestream_end - ptr) < mb_size)

            return -1;

        sl->intra_pcm_ptr = ptr;

        ptr += mb_size;



        ff_init_cabac_decoder(&sl->cabac, ptr, sl->cabac.bytestream_end - ptr);



        // All blocks are present

        h->cbp_table[mb_xy] = 0xf7ef;

        h->chroma_pred_mode_table[mb_xy] = 0;

        // In deblocking, the quantizer is 0

        h->cur_pic.qscale_table[mb_xy] = 0;

        // All coeffs are present

        memset(h->non_zero_count[mb_xy], 16, 48);

        h->cur_pic.mb_type[mb_xy] = mb_type;

        sl->last_qscale_diff = 0;

        return 0;

    }



    fill_decode_caches(h, sl, mb_type);



    if( IS_INTRA( mb_type ) ) {

        int i, pred_mode;

        if( IS_INTRA4x4( mb_type ) ) {

            if (dct8x8_allowed && get_cabac_noinline(&sl->cabac, &sl->cabac_state[399 + sl->neighbor_transform_size])) {

                mb_type |= MB_TYPE_8x8DCT;

                for( i = 0; i < 16; i+=4 ) {

                    int pred = pred_intra_mode(h, sl, i);

                    int mode = decode_cabac_mb_intra4x4_pred_mode(sl, pred);

                    fill_rectangle(&sl->intra4x4_pred_mode_cache[scan8[i]], 2, 2, 8, mode, 1);

                }

            } else {

                for( i = 0; i < 16; i++ ) {

                    int pred = pred_intra_mode(h, sl, i);

                    sl->intra4x4_pred_mode_cache[scan8[i]] = decode_cabac_mb_intra4x4_pred_mode(sl, pred);



                    ff_tlog(h->avctx, "i4x4 pred=%d mode=%d\n", pred,

                            sl->intra4x4_pred_mode_cache[scan8[i]]);

                }

            }

            write_back_intra_pred_mode(h, sl);

            if (ff_h264_check_intra4x4_pred_mode(h, sl) < 0 ) return -1;

        } else {

            sl->intra16x16_pred_mode = ff_h264_check_intra_pred_mode(h, sl, sl->intra16x16_pred_mode, 0);

            if (sl->intra16x16_pred_mode < 0) return -1;

        }

        if(decode_chroma){

            h->chroma_pred_mode_table[mb_xy] =

            pred_mode                        = decode_cabac_mb_chroma_pre_mode(h, sl);



            pred_mode= ff_h264_check_intra_pred_mode(h, sl, pred_mode, 1 );

            if( pred_mode < 0 ) return -1;

            sl->chroma_pred_mode = pred_mode;

        } else {

            sl->chroma_pred_mode = DC_128_PRED8x8;

        }

    } else if( partition_count == 4 ) {

        int i, j, sub_partition_count[4], list, ref[2][4];



        if (sl->slice_type_nos == AV_PICTURE_TYPE_B ) {

            for( i = 0; i < 4; i++ ) {

                sl->sub_mb_type[i] = decode_cabac_b_mb_sub_type(sl);

                sub_partition_count[i] = b_sub_mb_type_info[sl->sub_mb_type[i]].partition_count;

                sl->sub_mb_type[i]     = b_sub_mb_type_info[sl->sub_mb_type[i]].type;

            }

            if (IS_DIRECT(sl->sub_mb_type[0] | sl->sub_mb_type[1] |

                          sl->sub_mb_type[2] | sl->sub_mb_type[3])) {

                ff_h264_pred_direct_motion(h, sl, &mb_type);

                sl->ref_cache[0][scan8[4]] =

                sl->ref_cache[1][scan8[4]] =

                sl->ref_cache[0][scan8[12]] =

                sl->ref_cache[1][scan8[12]] = PART_NOT_AVAILABLE;

                    for( i = 0; i < 4; i++ )

                        fill_rectangle(&sl->direct_cache[scan8[4*i]], 2, 2, 8, (sl->sub_mb_type[i] >> 1) & 0xFF, 1);

            }

        } else {

            for( i = 0; i < 4; i++ ) {

                sl->sub_mb_type[i] = decode_cabac_p_mb_sub_type(sl);

                sub_partition_count[i] = p_sub_mb_type_info[sl->sub_mb_type[i]].partition_count;

                sl->sub_mb_type[i]     = p_sub_mb_type_info[sl->sub_mb_type[i]].type;

            }

        }



        for( list = 0; list < sl->list_count; list++ ) {

                for( i = 0; i < 4; i++ ) {

                    if(IS_DIRECT(sl->sub_mb_type[i])) continue;

                    if(IS_DIR(sl->sub_mb_type[i], 0, list)){

                        unsigned rc = sl->ref_count[list] << MB_MBAFF(sl);

                        if (rc > 1) {

                            ref[list][i] = decode_cabac_mb_ref(sl, list, 4 * i);

                            if (ref[list][i] >= rc) {

                                av_log(h->avctx, AV_LOG_ERROR, "Reference %d >= %d\n", ref[list][i], rc);

                                return -1;

                            }

                        }else

                            ref[list][i] = 0;

                    } else {

                        ref[list][i] = -1;

                    }

                    sl->ref_cache[list][scan8[4 * i] + 1] =

                    sl->ref_cache[list][scan8[4 * i] + 8] = sl->ref_cache[list][scan8[4 * i] + 9] = ref[list][i];

                }

        }



        if(dct8x8_allowed)

            dct8x8_allowed = get_dct8x8_allowed(h, sl);



        for (list = 0; list < sl->list_count; list++) {

            for(i=0; i<4; i++){

                sl->ref_cache[list][scan8[4 * i]] = sl->ref_cache[list][scan8[4 * i] + 1];

                if(IS_DIRECT(sl->sub_mb_type[i])){

                    fill_rectangle(sl->mvd_cache[list][scan8[4*i]], 2, 2, 8, 0, 2);

                    continue;

                }



                if(IS_DIR(sl->sub_mb_type[i], 0, list) && !IS_DIRECT(sl->sub_mb_type[i])){

                    const int sub_mb_type= sl->sub_mb_type[i];

                    const int block_width= (sub_mb_type & (MB_TYPE_16x16|MB_TYPE_16x8)) ? 2 : 1;

                    for(j=0; j<sub_partition_count[i]; j++){

                        int mpx, mpy;

                        int mx, my;

                        const int index= 4*i + block_width*j;

                        int16_t (* mv_cache)[2] = &sl->mv_cache[list][ scan8[index] ];

                        uint8_t (* mvd_cache)[2]= &sl->mvd_cache[list][ scan8[index] ];

                        pred_motion(h, sl, index, block_width, list, sl->ref_cache[list][ scan8[index] ], &mx, &my);

                        DECODE_CABAC_MB_MVD(sl, list, index)

                        ff_tlog(h->avctx, "final mv:%d %d\n", mx, my);



                        if(IS_SUB_8X8(sub_mb_type)){

                            mv_cache[ 1 ][0]=

                            mv_cache[ 8 ][0]= mv_cache[ 9 ][0]= mx;

                            mv_cache[ 1 ][1]=

                            mv_cache[ 8 ][1]= mv_cache[ 9 ][1]= my;



                            mvd_cache[ 1 ][0]=

                            mvd_cache[ 8 ][0]= mvd_cache[ 9 ][0]= mpx;

                            mvd_cache[ 1 ][1]=

                            mvd_cache[ 8 ][1]= mvd_cache[ 9 ][1]= mpy;

                        }else if(IS_SUB_8X4(sub_mb_type)){

                            mv_cache[ 1 ][0]= mx;

                            mv_cache[ 1 ][1]= my;



                            mvd_cache[ 1 ][0]=  mpx;

                            mvd_cache[ 1 ][1]= mpy;

                        }else if(IS_SUB_4X8(sub_mb_type)){

                            mv_cache[ 8 ][0]= mx;

                            mv_cache[ 8 ][1]= my;



                            mvd_cache[ 8 ][0]= mpx;

                            mvd_cache[ 8 ][1]= mpy;

                        }

                        mv_cache[ 0 ][0]= mx;

                        mv_cache[ 0 ][1]= my;



                        mvd_cache[ 0 ][0]= mpx;

                        mvd_cache[ 0 ][1]= mpy;

                    }

                }else{

                    fill_rectangle(sl->mv_cache [list][ scan8[4*i] ], 2, 2, 8, 0, 4);

                    fill_rectangle(sl->mvd_cache[list][ scan8[4*i] ], 2, 2, 8, 0, 2);

                }

            }

        }

    } else if( IS_DIRECT(mb_type) ) {

        ff_h264_pred_direct_motion(h, sl, &mb_type);

        fill_rectangle(sl->mvd_cache[0][scan8[0]], 4, 4, 8, 0, 2);

        fill_rectangle(sl->mvd_cache[1][scan8[0]], 4, 4, 8, 0, 2);

        dct8x8_allowed &= h->sps.direct_8x8_inference_flag;

    } else {

        int list, i;

        if(IS_16X16(mb_type)){

            for (list = 0; list < sl->list_count; list++) {

                if(IS_DIR(mb_type, 0, list)){

                    int ref;

                    unsigned rc = sl->ref_count[list] << MB_MBAFF(sl);

                    if (rc > 1) {

                        ref= decode_cabac_mb_ref(sl, list, 0);

                        if (ref >= rc) {

                            av_log(h->avctx, AV_LOG_ERROR, "Reference %d >= %d\n", ref, rc);

                            return -1;

                        }

                    }else

                        ref=0;

                    fill_rectangle(&sl->ref_cache[list][ scan8[0] ], 4, 4, 8, ref, 1);

                }

            }

            for (list = 0; list < sl->list_count; list++) {

                if(IS_DIR(mb_type, 0, list)){

                    int mx,my,mpx,mpy;

                    pred_motion(h, sl, 0, 4, list, sl->ref_cache[list][ scan8[0] ], &mx, &my);

                    DECODE_CABAC_MB_MVD(sl, list, 0)

                    ff_tlog(h->avctx, "final mv:%d %d\n", mx, my);



                    fill_rectangle(sl->mvd_cache[list][ scan8[0] ], 4, 4, 8, pack8to16(mpx,mpy), 2);

                    fill_rectangle(sl->mv_cache[list][ scan8[0] ], 4, 4, 8, pack16to32(mx,my), 4);

                }

            }

        }

        else if(IS_16X8(mb_type)){

            for (list = 0; list < sl->list_count; list++) {

                    for(i=0; i<2; i++){

                        if(IS_DIR(mb_type, i, list)){

                            int ref;

                            unsigned rc = sl->ref_count[list] << MB_MBAFF(sl);

                            if (rc > 1) {

                                ref= decode_cabac_mb_ref(sl, list, 8 * i);

                                if (ref >= rc) {

                                    av_log(h->avctx, AV_LOG_ERROR, "Reference %d >= %d\n", ref, rc);

                                    return -1;

                                }

                            }else

                                ref=0;

                            fill_rectangle(&sl->ref_cache[list][ scan8[0] + 16*i ], 4, 2, 8, ref, 1);

                        }else

                            fill_rectangle(&sl->ref_cache[list][ scan8[0] + 16*i ], 4, 2, 8, (LIST_NOT_USED&0xFF), 1);

                    }

            }

            for (list = 0; list < sl->list_count; list++) {

                for(i=0; i<2; i++){

                    if(IS_DIR(mb_type, i, list)){

                        int mx,my,mpx,mpy;

                        pred_16x8_motion(h, sl, 8*i, list, sl->ref_cache[list][scan8[0] + 16*i], &mx, &my);

                        DECODE_CABAC_MB_MVD(sl, list, 8*i)

                        ff_tlog(h->avctx, "final mv:%d %d\n", mx, my);



                        fill_rectangle(sl->mvd_cache[list][ scan8[0] + 16*i ], 4, 2, 8, pack8to16(mpx,mpy), 2);

                        fill_rectangle(sl->mv_cache[list][ scan8[0] + 16*i ], 4, 2, 8, pack16to32(mx,my), 4);

                    }else{

                        fill_rectangle(sl->mvd_cache[list][ scan8[0] + 16*i ], 4, 2, 8, 0, 2);

                        fill_rectangle(sl->mv_cache[list][ scan8[0] + 16*i ], 4, 2, 8, 0, 4);

                    }

                }

            }

        }else{

            av_assert2(IS_8X16(mb_type));

            for (list = 0; list < sl->list_count; list++) {

                    for(i=0; i<2; i++){

                        if(IS_DIR(mb_type, i, list)){ //FIXME optimize

                            int ref;

                            unsigned rc = sl->ref_count[list] << MB_MBAFF(sl);

                            if (rc > 1) {

                                ref = decode_cabac_mb_ref(sl, list, 4 * i);

                                if (ref >= rc) {

                                    av_log(h->avctx, AV_LOG_ERROR, "Reference %d >= %d\n", ref, rc);

                                    return -1;

                                }

                            }else

                                ref=0;

                            fill_rectangle(&sl->ref_cache[list][ scan8[0] + 2*i ], 2, 4, 8, ref, 1);

                        }else

                            fill_rectangle(&sl->ref_cache[list][ scan8[0] + 2*i ], 2, 4, 8, (LIST_NOT_USED&0xFF), 1);

                    }

            }

            for (list = 0; list < sl->list_count; list++) {

                for(i=0; i<2; i++){

                    if(IS_DIR(mb_type, i, list)){

                        int mx,my,mpx,mpy;

                        pred_8x16_motion(h, sl, i*4, list, sl->ref_cache[list][ scan8[0] + 2*i ], &mx, &my);

                        DECODE_CABAC_MB_MVD(sl, list, 4*i)



                        ff_tlog(h->avctx, "final mv:%d %d\n", mx, my);

                        fill_rectangle(sl->mvd_cache[list][ scan8[0] + 2*i ], 2, 4, 8, pack8to16(mpx,mpy), 2);

                        fill_rectangle(sl->mv_cache[list][ scan8[0] + 2*i ], 2, 4, 8, pack16to32(mx,my), 4);

                    }else{

                        fill_rectangle(sl->mvd_cache[list][ scan8[0] + 2*i ], 2, 4, 8, 0, 2);

                        fill_rectangle(sl->mv_cache[list][ scan8[0] + 2*i ], 2, 4, 8, 0, 4);

                    }

                }

            }

        }

    }



   if( IS_INTER( mb_type ) ) {

        h->chroma_pred_mode_table[mb_xy] = 0;

        write_back_motion(h, sl, mb_type);

   }



    if( !IS_INTRA16x16( mb_type ) ) {

        cbp  = decode_cabac_mb_cbp_luma(sl);

        if(decode_chroma)

            cbp |= decode_cabac_mb_cbp_chroma(sl) << 4;

    } else {

        if (!decode_chroma && cbp>15) {

            av_log(h->avctx, AV_LOG_ERROR, "gray chroma\n");

            return AVERROR_INVALIDDATA;

        }

    }



    h->cbp_table[mb_xy] = sl->cbp = cbp;



    if( dct8x8_allowed && (cbp&15) && !IS_INTRA( mb_type ) ) {

        mb_type |= MB_TYPE_8x8DCT * get_cabac_noinline(&sl->cabac, &sl->cabac_state[399 + sl->neighbor_transform_size]);

    }



    /* It would be better to do this in fill_decode_caches, but we don't know

     * the transform mode of the current macroblock there. */

    if (CHROMA444(h) && IS_8x8DCT(mb_type)){

        int i;

        uint8_t *nnz_cache = sl->non_zero_count_cache;

        for (i = 0; i < 2; i++){

            if (sl->left_type[LEFT(i)] && !IS_8x8DCT(sl->left_type[LEFT(i)])) {

                nnz_cache[3+8* 1 + 2*8*i]=

                nnz_cache[3+8* 2 + 2*8*i]=

                nnz_cache[3+8* 6 + 2*8*i]=

                nnz_cache[3+8* 7 + 2*8*i]=

                nnz_cache[3+8*11 + 2*8*i]=

                nnz_cache[3+8*12 + 2*8*i]= IS_INTRA(mb_type) ? 64 : 0;

            }

        }

        if (sl->top_type && !IS_8x8DCT(sl->top_type)){

            uint32_t top_empty = CABAC(h) && !IS_INTRA(mb_type) ? 0 : 0x40404040;

            AV_WN32A(&nnz_cache[4+8* 0], top_empty);

            AV_WN32A(&nnz_cache[4+8* 5], top_empty);

            AV_WN32A(&nnz_cache[4+8*10], top_empty);

        }

    }

    h->cur_pic.mb_type[mb_xy] = mb_type;



    if( cbp || IS_INTRA16x16( mb_type ) ) {

        const uint8_t *scan, *scan8x8;

        const uint32_t *qmul;



        if(IS_INTERLACED(mb_type)){

            scan8x8 = sl->qscale ? h->field_scan8x8 : h->field_scan8x8_q0;

            scan    = sl->qscale ? h->field_scan : h->field_scan_q0;

        }else{

            scan8x8 = sl->qscale ? h->zigzag_scan8x8 : h->zigzag_scan8x8_q0;

            scan    = sl->qscale ? h->zigzag_scan : h->zigzag_scan_q0;

        }



        // decode_cabac_mb_dqp

        if(get_cabac_noinline( &sl->cabac, &sl->cabac_state[60 + (sl->last_qscale_diff != 0)])){

            int val = 1;

            int ctx= 2;

            const int max_qp = 51 + 6*(h->sps.bit_depth_luma-8);



            while( get_cabac_noinline( &sl->cabac, &sl->cabac_state[60 + ctx] ) ) {

                ctx= 3;

                val++;

                if(val > 2*max_qp){ //prevent infinite loop

                    av_log(h->avctx, AV_LOG_ERROR, "cabac decode of qscale diff failed at %d %d\n", sl->mb_x, sl->mb_y);

                    return -1;

                }

            }



            if( val&0x01 )

                val=   (val + 1)>>1 ;

            else

                val= -((val + 1)>>1);

            sl->last_qscale_diff = val;

            sl->qscale += val;

            if (((unsigned)sl->qscale) > max_qp){

                if (sl->qscale < 0) sl->qscale += max_qp + 1;

                else                sl->qscale -= max_qp + 1;

            }

            sl->chroma_qp[0] = get_chroma_qp(h, 0, sl->qscale);

            sl->chroma_qp[1] = get_chroma_qp(h, 1, sl->qscale);

        }else

            sl->last_qscale_diff=0;



        decode_cabac_luma_residual(h, sl, scan, scan8x8, pixel_shift, mb_type, cbp, 0);

        if (CHROMA444(h)) {

            decode_cabac_luma_residual(h, sl, scan, scan8x8, pixel_shift, mb_type, cbp, 1);

            decode_cabac_luma_residual(h, sl, scan, scan8x8, pixel_shift, mb_type, cbp, 2);

        } else if (CHROMA422(h)) {

            if( cbp&0x30 ){

                int c;

                for (c = 0; c < 2; c++)

                    decode_cabac_residual_dc_422(h, sl, sl->mb + ((256 + 16*16*c) << pixel_shift), 3,

                                                 CHROMA_DC_BLOCK_INDEX + c,

                                                 chroma422_dc_scan, 8);

            }



            if( cbp&0x20 ) {

                int c, i, i8x8;

                for( c = 0; c < 2; c++ ) {

                    int16_t *mb = sl->mb + (16*(16 + 16*c) << pixel_shift);

                    qmul = h->dequant4_coeff[c+1+(IS_INTRA( mb_type ) ? 0:3)][sl->chroma_qp[c]];

                    for (i8x8 = 0; i8x8 < 2; i8x8++) {

                        for (i = 0; i < 4; i++) {

                            const int index = 16 + 16 * c + 8*i8x8 + i;

                            decode_cabac_residual_nondc(h, sl, mb, 4, index, scan + 1, qmul, 15);

                            mb += 16<<pixel_shift;

                        }

                    }

                }

            } else {

                fill_rectangle(&sl->non_zero_count_cache[scan8[16]], 4, 4, 8, 0, 1);

                fill_rectangle(&sl->non_zero_count_cache[scan8[32]], 4, 4, 8, 0, 1);

            }

        } else /* yuv420 */ {

            if( cbp&0x30 ){

                int c;

                for (c = 0; c < 2; c++)

                    decode_cabac_residual_dc(h, sl, sl->mb + ((256 + 16*16*c) << pixel_shift), 3, CHROMA_DC_BLOCK_INDEX+c, chroma_dc_scan, 4);

            }



            if( cbp&0x20 ) {

                int c, i;

                for( c = 0; c < 2; c++ ) {

                    qmul = h->dequant4_coeff[c+1+(IS_INTRA( mb_type ) ? 0:3)][sl->chroma_qp[c]];

                    for( i = 0; i < 4; i++ ) {

                        const int index = 16 + 16 * c + i;

                        decode_cabac_residual_nondc(h, sl, sl->mb + (16*index << pixel_shift), 4, index, scan + 1, qmul, 15);

                    }

                }

            } else {

                fill_rectangle(&sl->non_zero_count_cache[scan8[16]], 4, 4, 8, 0, 1);

                fill_rectangle(&sl->non_zero_count_cache[scan8[32]], 4, 4, 8, 0, 1);

            }

        }

    } else {

        fill_rectangle(&sl->non_zero_count_cache[scan8[ 0]], 4, 4, 8, 0, 1);

        fill_rectangle(&sl->non_zero_count_cache[scan8[16]], 4, 4, 8, 0, 1);

        fill_rectangle(&sl->non_zero_count_cache[scan8[32]], 4, 4, 8, 0, 1);

        sl->last_qscale_diff = 0;

    }



    h->cur_pic.qscale_table[mb_xy] = sl->qscale;

    write_back_non_zero_count(h, sl);



    return 0;

}
