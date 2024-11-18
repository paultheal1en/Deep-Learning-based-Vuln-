static av_always_inline void filter_mb_dir(H264Context *h, int mb_x, int mb_y, uint8_t *img_y, uint8_t *img_cb, uint8_t *img_cr, unsigned int linesize, unsigned int uvlinesize, int mb_xy, int mb_type, int mvy_limit, int first_vertical_edge_done, int chroma, int chroma444, int dir) {

    MpegEncContext * const s = &h->s;

    int edge;

    int chroma_qp_avg[2];

    const int mbm_xy = dir == 0 ? mb_xy -1 : h->top_mb_xy;

    const int mbm_type = dir == 0 ? h->left_type[LTOP] : h->top_type;



    // how often to recheck mv-based bS when iterating between edges

    static const uint8_t mask_edge_tab[2][8]={{0,3,3,3,1,1,1,1},

                                              {0,3,1,1,3,3,3,3}};

    const int mask_edge = mask_edge_tab[dir][(mb_type>>3)&7];

    const int edges = mask_edge== 3 && !(h->cbp&15) ? 1 : 4;



    // how often to recheck mv-based bS when iterating along each edge

    const int mask_par0 = mb_type & (MB_TYPE_16x16 | (MB_TYPE_8x16 >> dir));



    if(mbm_type && !first_vertical_edge_done){



        if (FRAME_MBAFF && (dir == 1) && ((mb_y&1) == 0)

            && IS_INTERLACED(mbm_type&~mb_type)

            ) {

            // This is a special case in the norm where the filtering must

            // be done twice (one each of the field) even if we are in a

            // frame macroblock.

            //

            unsigned int tmp_linesize   = 2 *   linesize;

            unsigned int tmp_uvlinesize = 2 * uvlinesize;

            int mbn_xy = mb_xy - 2 * s->mb_stride;

            int j;



            for(j=0; j<2; j++, mbn_xy += s->mb_stride){

                DECLARE_ALIGNED(8, int16_t, bS)[4];

                int qp;

                if (IS_INTRA(mb_type | s->current_picture.f.mb_type[mbn_xy])) {

                    AV_WN64A(bS, 0x0003000300030003ULL);

                } else {

                    if (!CABAC && IS_8x8DCT(s->current_picture.f.mb_type[mbn_xy])) {

                        bS[0]= 1+((h->cbp_table[mbn_xy] & 0x4000)||h->non_zero_count_cache[scan8[0]+0]);

                        bS[1]= 1+((h->cbp_table[mbn_xy] & 0x4000)||h->non_zero_count_cache[scan8[0]+1]);

                        bS[2]= 1+((h->cbp_table[mbn_xy] & 0x8000)||h->non_zero_count_cache[scan8[0]+2]);

                        bS[3]= 1+((h->cbp_table[mbn_xy] & 0x8000)||h->non_zero_count_cache[scan8[0]+3]);

                    }else{

                    const uint8_t *mbn_nnz = h->non_zero_count[mbn_xy] + 3*4;

                    int i;

                    for( i = 0; i < 4; i++ ) {

                        bS[i] = 1 + !!(h->non_zero_count_cache[scan8[0]+i] | mbn_nnz[i]);

                    }

                    }

                }

                // Do not use s->qscale as luma quantizer because it has not the same

                // value in IPCM macroblocks.

                qp = (s->current_picture.f.qscale_table[mb_xy] + s->current_picture.f.qscale_table[mbn_xy] + 1) >> 1;

                tprintf(s->avctx, "filter mb:%d/%d dir:%d edge:%d, QPy:%d ls:%d uvls:%d", mb_x, mb_y, dir, edge, qp, tmp_linesize, tmp_uvlinesize);

                { int i; for (i = 0; i < 4; i++) tprintf(s->avctx, " bS[%d]:%d", i, bS[i]); tprintf(s->avctx, "\n"); }

                filter_mb_edgeh( &img_y[j*linesize], tmp_linesize, bS, qp, h );

                chroma_qp_avg[0] = (h->chroma_qp[0] + get_chroma_qp(h, 0, s->current_picture.f.qscale_table[mbn_xy]) + 1) >> 1;

                chroma_qp_avg[1] = (h->chroma_qp[1] + get_chroma_qp(h, 1, s->current_picture.f.qscale_table[mbn_xy]) + 1) >> 1;

                if (chroma) {

                    if (chroma444) {

                        filter_mb_edgeh (&img_cb[j*uvlinesize], tmp_uvlinesize, bS, chroma_qp_avg[0], h);

                        filter_mb_edgeh (&img_cr[j*uvlinesize], tmp_uvlinesize, bS, chroma_qp_avg[1], h);

                    } else {

                        filter_mb_edgech(&img_cb[j*uvlinesize], tmp_uvlinesize, bS, chroma_qp_avg[0], h);

                        filter_mb_edgech(&img_cr[j*uvlinesize], tmp_uvlinesize, bS, chroma_qp_avg[1], h);

                    }

                }

            }

        }else{

            DECLARE_ALIGNED(8, int16_t, bS)[4];

            int qp;



            if( IS_INTRA(mb_type|mbm_type)) {

                AV_WN64A(bS, 0x0003000300030003ULL);

                if (   (!IS_INTERLACED(mb_type|mbm_type))

                    || ((FRAME_MBAFF || (s->picture_structure != PICT_FRAME)) && (dir == 0))

                )

                    AV_WN64A(bS, 0x0004000400040004ULL);

            } else {

                int i;

                int mv_done;



                if( dir && FRAME_MBAFF && IS_INTERLACED(mb_type ^ mbm_type)) {

                    AV_WN64A(bS, 0x0001000100010001ULL);

                    mv_done = 1;

                }

                else if( mask_par0 && ((mbm_type & (MB_TYPE_16x16 | (MB_TYPE_8x16 >> dir)))) ) {

                    int b_idx= 8 + 4;

                    int bn_idx= b_idx - (dir ? 8:1);



                    bS[0] = bS[1] = bS[2] = bS[3] = check_mv(h, 8 + 4, bn_idx, mvy_limit);

                    mv_done = 1;

                }

                else

                    mv_done = 0;



                for( i = 0; i < 4; i++ ) {

                    int x = dir == 0 ? 0 : i;

                    int y = dir == 0 ? i    : 0;

                    int b_idx= 8 + 4 + x + 8*y;

                    int bn_idx= b_idx - (dir ? 8:1);



                    if( h->non_zero_count_cache[b_idx] |

                        h->non_zero_count_cache[bn_idx] ) {

                        bS[i] = 2;

                    }

                    else if(!mv_done)

                    {

                        bS[i] = check_mv(h, b_idx, bn_idx, mvy_limit);

                    }

                }

            }



            /* Filter edge */

            // Do not use s->qscale as luma quantizer because it has not the same

            // value in IPCM macroblocks.

            if(bS[0]+bS[1]+bS[2]+bS[3]){

                qp = (s->current_picture.f.qscale_table[mb_xy] + s->current_picture.f.qscale_table[mbm_xy] + 1) >> 1;

                //tprintf(s->avctx, "filter mb:%d/%d dir:%d edge:%d, QPy:%d, QPc:%d, QPcn:%d\n", mb_x, mb_y, dir, edge, qp, h->chroma_qp[0], s->current_picture.qscale_table[mbn_xy]);

                tprintf(s->avctx, "filter mb:%d/%d dir:%d edge:%d, QPy:%d ls:%d uvls:%d", mb_x, mb_y, dir, edge, qp, linesize, uvlinesize);

                //{ int i; for (i = 0; i < 4; i++) tprintf(s->avctx, " bS[%d]:%d", i, bS[i]); tprintf(s->avctx, "\n"); }

                chroma_qp_avg[0] = (h->chroma_qp[0] + get_chroma_qp(h, 0, s->current_picture.f.qscale_table[mbm_xy]) + 1) >> 1;

                chroma_qp_avg[1] = (h->chroma_qp[1] + get_chroma_qp(h, 1, s->current_picture.f.qscale_table[mbm_xy]) + 1) >> 1;

                if( dir == 0 ) {

                    filter_mb_edgev( &img_y[0], linesize, bS, qp, h );

                    if (chroma) {

                        if (chroma444) {

                            filter_mb_edgev ( &img_cb[0], uvlinesize, bS, chroma_qp_avg[0], h);

                            filter_mb_edgev ( &img_cr[0], uvlinesize, bS, chroma_qp_avg[1], h);

                        } else {

                            filter_mb_edgecv( &img_cb[0], uvlinesize, bS, chroma_qp_avg[0], h);

                            filter_mb_edgecv( &img_cr[0], uvlinesize, bS, chroma_qp_avg[1], h);

                        }

                    }

                } else {

                    filter_mb_edgeh( &img_y[0], linesize, bS, qp, h );

                    if (chroma) {

                        if (chroma444) {

                            filter_mb_edgeh ( &img_cb[0], uvlinesize, bS, chroma_qp_avg[0], h);

                            filter_mb_edgeh ( &img_cr[0], uvlinesize, bS, chroma_qp_avg[1], h);

                        } else {

                            filter_mb_edgech( &img_cb[0], uvlinesize, bS, chroma_qp_avg[0], h);

                            filter_mb_edgech( &img_cr[0], uvlinesize, bS, chroma_qp_avg[1], h);

                        }

                    }

                }

            }

        }

    }



    /* Calculate bS */

    for( edge = 1; edge < edges; edge++ ) {

        DECLARE_ALIGNED(8, int16_t, bS)[4];

        int qp;



        if( IS_8x8DCT(mb_type & (edge<<24)) ) // (edge&1) && IS_8x8DCT(mb_type)

            continue;



        if( IS_INTRA(mb_type)) {

            AV_WN64A(bS, 0x0003000300030003ULL);

        } else {

            int i;

            int mv_done;



            if( edge & mask_edge ) {

                AV_ZERO64(bS);

                mv_done = 1;

            }

            else if( mask_par0 ) {

                int b_idx= 8 + 4 + edge * (dir ? 8:1);

                int bn_idx= b_idx - (dir ? 8:1);



                bS[0] = bS[1] = bS[2] = bS[3] = check_mv(h, b_idx, bn_idx, mvy_limit);

                mv_done = 1;

            }

            else

                mv_done = 0;



            for( i = 0; i < 4; i++ ) {

                int x = dir == 0 ? edge : i;

                int y = dir == 0 ? i    : edge;

                int b_idx= 8 + 4 + x + 8*y;

                int bn_idx= b_idx - (dir ? 8:1);



                if( h->non_zero_count_cache[b_idx] |

                    h->non_zero_count_cache[bn_idx] ) {

                    bS[i] = 2;

                }

                else if(!mv_done)

                {

                    bS[i] = check_mv(h, b_idx, bn_idx, mvy_limit);

                }

            }



            if(bS[0]+bS[1]+bS[2]+bS[3] == 0)

                continue;

        }



        /* Filter edge */

        // Do not use s->qscale as luma quantizer because it has not the same

        // value in IPCM macroblocks.

        qp = s->current_picture.f.qscale_table[mb_xy];

        //tprintf(s->avctx, "filter mb:%d/%d dir:%d edge:%d, QPy:%d, QPc:%d, QPcn:%d\n", mb_x, mb_y, dir, edge, qp, h->chroma_qp[0], s->current_picture.qscale_table[mbn_xy]);

        tprintf(s->avctx, "filter mb:%d/%d dir:%d edge:%d, QPy:%d ls:%d uvls:%d", mb_x, mb_y, dir, edge, qp, linesize, uvlinesize);

        //{ int i; for (i = 0; i < 4; i++) tprintf(s->avctx, " bS[%d]:%d", i, bS[i]); tprintf(s->avctx, "\n"); }

        if( dir == 0 ) {

            filter_mb_edgev( &img_y[4*edge << h->pixel_shift], linesize, bS, qp, h );

            if (chroma) {

                if (chroma444) {

                    filter_mb_edgev ( &img_cb[4*edge << h->pixel_shift], uvlinesize, bS, h->chroma_qp[0], h);

                    filter_mb_edgev ( &img_cr[4*edge << h->pixel_shift], uvlinesize, bS, h->chroma_qp[1], h);

                } else if( (edge&1) == 0 ) {

                    filter_mb_edgecv( &img_cb[2*edge << h->pixel_shift], uvlinesize, bS, h->chroma_qp[0], h);

                    filter_mb_edgecv( &img_cr[2*edge << h->pixel_shift], uvlinesize, bS, h->chroma_qp[1], h);

                }

            }

        } else {

            filter_mb_edgeh( &img_y[4*edge*linesize], linesize, bS, qp, h );

            if (chroma) {

                if (chroma444) {

                    filter_mb_edgeh ( &img_cb[4*edge*uvlinesize], uvlinesize, bS, h->chroma_qp[0], h);

                    filter_mb_edgeh ( &img_cr[4*edge*uvlinesize], uvlinesize, bS, h->chroma_qp[1], h);

                } else if( (edge&1) == 0 ) {

                    filter_mb_edgech( &img_cb[2*edge*uvlinesize], uvlinesize, bS, h->chroma_qp[0], h);

                    filter_mb_edgech( &img_cr[2*edge*uvlinesize], uvlinesize, bS, h->chroma_qp[1], h);

                }

            }

        }

    }

}
