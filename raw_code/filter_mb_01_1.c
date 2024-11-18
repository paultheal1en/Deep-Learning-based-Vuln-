static void filter_mb( H264Context *h, int mb_x, int mb_y, uint8_t *img_y, uint8_t *img_cb, uint8_t *img_cr, unsigned int linesize, unsigned int uvlinesize) {

    MpegEncContext * const s = &h->s;

    const int mb_xy= mb_x + mb_y*s->mb_stride;

    const int mb_type = s->current_picture.mb_type[mb_xy];

    const int mvy_limit = IS_INTERLACED(mb_type) ? 2 : 4;

    int first_vertical_edge_done = 0;

    int dir;

    /* FIXME: A given frame may occupy more than one position in

     * the reference list. So ref2frm should be populated with

     * frame numbers, not indices. */

    static const int ref2frm[34] = {-1,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,

                                    16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};



    //for sufficiently low qp, filtering wouldn't do anything

    //this is a conservative estimate: could also check beta_offset and more accurate chroma_qp

    if(!FRAME_MBAFF){

        int qp_thresh = 15 - h->slice_alpha_c0_offset - FFMAX(0, h->pps.chroma_qp_index_offset);

        int qp = s->current_picture.qscale_table[mb_xy];

        if(qp <= qp_thresh

           && (mb_x == 0 || ((qp + s->current_picture.qscale_table[mb_xy-1] + 1)>>1) <= qp_thresh)

           && (mb_y == 0 || ((qp + s->current_picture.qscale_table[h->top_mb_xy] + 1)>>1) <= qp_thresh)){

            return;

        }

    }



    if (FRAME_MBAFF

            // left mb is in picture

            && h->slice_table[mb_xy-1] != 255

            // and current and left pair do not have the same interlaced type

            && (IS_INTERLACED(mb_type) != IS_INTERLACED(s->current_picture.mb_type[mb_xy-1]))

            // and left mb is in the same slice if deblocking_filter == 2

            && (h->deblocking_filter!=2 || h->slice_table[mb_xy-1] == h->slice_table[mb_xy])) {

        /* First vertical edge is different in MBAFF frames

         * There are 8 different bS to compute and 2 different Qp

         */

        const int pair_xy = mb_x + (mb_y&~1)*s->mb_stride;

        const int left_mb_xy[2] = { pair_xy-1, pair_xy-1+s->mb_stride };

        int16_t bS[8];

        int qp[2];

        int chroma_qp[2];

        int mb_qp, mbn0_qp, mbn1_qp;

        int i;

        first_vertical_edge_done = 1;



        if( IS_INTRA(mb_type) )

            bS[0] = bS[1] = bS[2] = bS[3] = bS[4] = bS[5] = bS[6] = bS[7] = 4;

        else {

            for( i = 0; i < 8; i++ ) {

                int mbn_xy = MB_FIELD ? left_mb_xy[i>>2] : left_mb_xy[i&1];



                if( IS_INTRA( s->current_picture.mb_type[mbn_xy] ) )

                    bS[i] = 4;

                else if( h->non_zero_count_cache[12+8*(i>>1)] != 0 ||

                         /* FIXME: with 8x8dct + cavlc, should check cbp instead of nnz */

                         h->non_zero_count[mbn_xy][MB_FIELD ? i&3 : (i>>2)+(mb_y&1)*2] )

                    bS[i] = 2;

                else

                    bS[i] = 1;

            }

        }



        mb_qp = s->current_picture.qscale_table[mb_xy];

        mbn0_qp = s->current_picture.qscale_table[left_mb_xy[0]];

        mbn1_qp = s->current_picture.qscale_table[left_mb_xy[1]];

        qp[0] = ( mb_qp + mbn0_qp + 1 ) >> 1;

        chroma_qp[0] = ( get_chroma_qp( h, mb_qp ) +

                         get_chroma_qp( h, mbn0_qp ) + 1 ) >> 1;

        qp[1] = ( mb_qp + mbn1_qp + 1 ) >> 1;

        chroma_qp[1] = ( get_chroma_qp( h, mb_qp ) +

                         get_chroma_qp( h, mbn1_qp ) + 1 ) >> 1;



        /* Filter edge */

        tprintf(s->avctx, "filter mb:%d/%d MBAFF, QPy:%d/%d, QPc:%d/%d ls:%d uvls:%d", mb_x, mb_y, qp[0], qp[1], chroma_qp[0], chroma_qp[1], linesize, uvlinesize);

        { int i; for (i = 0; i < 8; i++) tprintf(s->avctx, " bS[%d]:%d", i, bS[i]); tprintf(s->avctx, "\n"); }

        filter_mb_mbaff_edgev ( h, &img_y [0], linesize,   bS, qp );

        filter_mb_mbaff_edgecv( h, &img_cb[0], uvlinesize, bS, chroma_qp );

        filter_mb_mbaff_edgecv( h, &img_cr[0], uvlinesize, bS, chroma_qp );

    }

    /* dir : 0 -> vertical edge, 1 -> horizontal edge */

    for( dir = 0; dir < 2; dir++ )

    {

        int edge;

        const int mbm_xy = dir == 0 ? mb_xy -1 : h->top_mb_xy;

        const int mbm_type = s->current_picture.mb_type[mbm_xy];

        int start = h->slice_table[mbm_xy] == 255 ? 1 : 0;



        const int edges = (mb_type & (MB_TYPE_16x16|MB_TYPE_SKIP))

                                  == (MB_TYPE_16x16|MB_TYPE_SKIP) ? 1 : 4;

        // how often to recheck mv-based bS when iterating between edges

        const int mask_edge = (mb_type & (MB_TYPE_16x16 | (MB_TYPE_16x8 << dir))) ? 3 :

                              (mb_type & (MB_TYPE_8x16 >> dir)) ? 1 : 0;

        // how often to recheck mv-based bS when iterating along each edge

        const int mask_par0 = mb_type & (MB_TYPE_16x16 | (MB_TYPE_8x16 >> dir));



        if (first_vertical_edge_done) {

            start = 1;

            first_vertical_edge_done = 0;

        }



        if (h->deblocking_filter==2 && h->slice_table[mbm_xy] != h->slice_table[mb_xy])

            start = 1;



        if (FRAME_MBAFF && (dir == 1) && ((mb_y&1) == 0) && start == 0

            && !IS_INTERLACED(mb_type)

            && IS_INTERLACED(mbm_type)

            ) {

            // This is a special case in the norm where the filtering must

            // be done twice (one each of the field) even if we are in a

            // frame macroblock.

            //

            static const int nnz_idx[4] = {4,5,6,3};

            unsigned int tmp_linesize   = 2 *   linesize;

            unsigned int tmp_uvlinesize = 2 * uvlinesize;

            int mbn_xy = mb_xy - 2 * s->mb_stride;

            int qp, chroma_qp;

            int i, j;

            int16_t bS[4];



            for(j=0; j<2; j++, mbn_xy += s->mb_stride){

                if( IS_INTRA(mb_type) ||

                    IS_INTRA(s->current_picture.mb_type[mbn_xy]) ) {

                    bS[0] = bS[1] = bS[2] = bS[3] = 3;

                } else {

                    const uint8_t *mbn_nnz = h->non_zero_count[mbn_xy];

                    for( i = 0; i < 4; i++ ) {

                        if( h->non_zero_count_cache[scan8[0]+i] != 0 ||

                            mbn_nnz[nnz_idx[i]] != 0 )

                            bS[i] = 2;

                        else

                            bS[i] = 1;

                    }

                }

                // Do not use s->qscale as luma quantizer because it has not the same

                // value in IPCM macroblocks.

                qp = ( s->current_picture.qscale_table[mb_xy] + s->current_picture.qscale_table[mbn_xy] + 1 ) >> 1;

                tprintf(s->avctx, "filter mb:%d/%d dir:%d edge:%d, QPy:%d ls:%d uvls:%d", mb_x, mb_y, dir, edge, qp, tmp_linesize, tmp_uvlinesize);

                { int i; for (i = 0; i < 4; i++) tprintf(s->avctx, " bS[%d]:%d", i, bS[i]); tprintf(s->avctx, "\n"); }

                filter_mb_edgeh( h, &img_y[j*linesize], tmp_linesize, bS, qp );

                chroma_qp = ( h->chroma_qp +

                              get_chroma_qp( h, s->current_picture.qscale_table[mbn_xy] ) + 1 ) >> 1;

                filter_mb_edgech( h, &img_cb[j*uvlinesize], tmp_uvlinesize, bS, chroma_qp );

                filter_mb_edgech( h, &img_cr[j*uvlinesize], tmp_uvlinesize, bS, chroma_qp );

            }



            start = 1;

        }



        /* Calculate bS */

        for( edge = start; edge < edges; edge++ ) {

            /* mbn_xy: neighbor macroblock */

            const int mbn_xy = edge > 0 ? mb_xy : mbm_xy;

            const int mbn_type = s->current_picture.mb_type[mbn_xy];

            int16_t bS[4];

            int qp;



            if( (edge&1) && IS_8x8DCT(mb_type) )

                continue;



            if( IS_INTRA(mb_type) ||

                IS_INTRA(mbn_type) ) {

                int value;

                if (edge == 0) {

                    if (   (!IS_INTERLACED(mb_type) && !IS_INTERLACED(mbm_type))

                        || ((FRAME_MBAFF || (s->picture_structure != PICT_FRAME)) && (dir == 0))

                    ) {

                        value = 4;

                    } else {

                        value = 3;

                    }

                } else {

                    value = 3;

                }

                bS[0] = bS[1] = bS[2] = bS[3] = value;

            } else {

                int i, l;

                int mv_done;



                if( edge & mask_edge ) {

                    bS[0] = bS[1] = bS[2] = bS[3] = 0;

                    mv_done = 1;

                }

                else if( FRAME_MBAFF && IS_INTERLACED(mb_type ^ mbn_type)) {

                    bS[0] = bS[1] = bS[2] = bS[3] = 1;

                    mv_done = 1;

                }

                else if( mask_par0 && (edge || (mbn_type & (MB_TYPE_16x16 | (MB_TYPE_8x16 >> dir)))) ) {

                    int b_idx= 8 + 4 + edge * (dir ? 8:1);

                    int bn_idx= b_idx - (dir ? 8:1);

                    int v = 0;

                    for( l = 0; !v && l < 1 + (h->slice_type == B_TYPE); l++ ) {

                        v |= ref2frm[h->ref_cache[l][b_idx]+2] != ref2frm[h->ref_cache[l][bn_idx]+2] ||

                             FFABS( h->mv_cache[l][b_idx][0] - h->mv_cache[l][bn_idx][0] ) >= 4 ||

                             FFABS( h->mv_cache[l][b_idx][1] - h->mv_cache[l][bn_idx][1] ) >= mvy_limit;

                    }

                    bS[0] = bS[1] = bS[2] = bS[3] = v;

                    mv_done = 1;

                }

                else

                    mv_done = 0;



                for( i = 0; i < 4; i++ ) {

                    int x = dir == 0 ? edge : i;

                    int y = dir == 0 ? i    : edge;

                    int b_idx= 8 + 4 + x + 8*y;

                    int bn_idx= b_idx - (dir ? 8:1);



                    if( h->non_zero_count_cache[b_idx] != 0 ||

                        h->non_zero_count_cache[bn_idx] != 0 ) {

                        bS[i] = 2;

                    }

                    else if(!mv_done)

                    {

                        bS[i] = 0;

                        for( l = 0; l < 1 + (h->slice_type == B_TYPE); l++ ) {

                            if( ref2frm[h->ref_cache[l][b_idx]+2] != ref2frm[h->ref_cache[l][bn_idx]+2] ||

                                FFABS( h->mv_cache[l][b_idx][0] - h->mv_cache[l][bn_idx][0] ) >= 4 ||

                                FFABS( h->mv_cache[l][b_idx][1] - h->mv_cache[l][bn_idx][1] ) >= mvy_limit ) {

                                bS[i] = 1;

                                break;

                            }

                        }

                    }

                }



                if(bS[0]+bS[1]+bS[2]+bS[3] == 0)

                    continue;

            }



            /* Filter edge */

            // Do not use s->qscale as luma quantizer because it has not the same

            // value in IPCM macroblocks.

            qp = ( s->current_picture.qscale_table[mb_xy] + s->current_picture.qscale_table[mbn_xy] + 1 ) >> 1;

            //tprintf(s->avctx, "filter mb:%d/%d dir:%d edge:%d, QPy:%d, QPc:%d, QPcn:%d\n", mb_x, mb_y, dir, edge, qp, h->chroma_qp, s->current_picture.qscale_table[mbn_xy]);

            tprintf(s->avctx, "filter mb:%d/%d dir:%d edge:%d, QPy:%d ls:%d uvls:%d", mb_x, mb_y, dir, edge, qp, linesize, uvlinesize);

            { int i; for (i = 0; i < 4; i++) tprintf(s->avctx, " bS[%d]:%d", i, bS[i]); tprintf(s->avctx, "\n"); }

            if( dir == 0 ) {

                filter_mb_edgev( h, &img_y[4*edge], linesize, bS, qp );

                if( (edge&1) == 0 ) {

                    int chroma_qp = ( h->chroma_qp +

                                      get_chroma_qp( h, s->current_picture.qscale_table[mbn_xy] ) + 1 ) >> 1;

                    filter_mb_edgecv( h, &img_cb[2*edge], uvlinesize, bS, chroma_qp );

                    filter_mb_edgecv( h, &img_cr[2*edge], uvlinesize, bS, chroma_qp );

                }

            } else {

                filter_mb_edgeh( h, &img_y[4*edge*linesize], linesize, bS, qp );

                if( (edge&1) == 0 ) {

                    int chroma_qp = ( h->chroma_qp +

                                      get_chroma_qp( h, s->current_picture.qscale_table[mbn_xy] ) + 1 ) >> 1;

                    filter_mb_edgech( h, &img_cb[2*edge*uvlinesize], uvlinesize, bS, chroma_qp );

                    filter_mb_edgech( h, &img_cr[2*edge*uvlinesize], uvlinesize, bS, chroma_qp );

                }

            }

        }

    }

}
