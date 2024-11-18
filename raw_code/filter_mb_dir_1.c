static av_always_inline void filter_mb_dir(const H264Context *h, H264SliceContext *sl,

                                           int mb_x, int mb_y,

                                           uint8_t *img_y, uint8_t *img_cb, uint8_t *img_cr,

                                           unsigned int linesize, unsigned int uvlinesize,

                                           int mb_xy, int mb_type, int mvy_limit,

                                           int first_vertical_edge_done, int a, int b,

                                           int chroma, int dir)

{

    int edge;

    int chroma_qp_avg[2];

    int chroma444 = CHROMA444(h);

    int chroma422 = CHROMA422(h);

    const int mbm_xy = dir == 0 ? mb_xy -1 : sl->top_mb_xy;

    const int mbm_type = dir == 0 ? sl->left_type[LTOP] : sl->top_type;



    // how often to recheck mv-based bS when iterating between edges

    static const uint8_t mask_edge_tab[2][8]={{0,3,3,3,1,1,1,1},

                                              {0,3,1,1,3,3,3,3}};

    const int mask_edge = mask_edge_tab[dir][(mb_type>>3)&7];

    const int edges = mask_edge== 3 && !(sl->cbp&15) ? 1 : 4;



    // how often to recheck mv-based bS when iterating along each edge

    const int mask_par0 = mb_type & (MB_TYPE_16x16 | (MB_TYPE_8x16 >> dir));



    if(mbm_type && !first_vertical_edge_done){



        if (FRAME_MBAFF(h) && (dir == 1) && ((mb_y&1) == 0)

            && IS_INTERLACED(mbm_type&~mb_type)

            ) {

            // This is a special case in the norm where the filtering must

            // be done twice (one each of the field) even if we are in a

            // frame macroblock.

            //

            unsigned int tmp_linesize   = 2 *   linesize;

            unsigned int tmp_uvlinesize = 2 * uvlinesize;

            int mbn_xy = mb_xy - 2 * h->mb_stride;

            int j;



            for(j=0; j<2; j++, mbn_xy += h->mb_stride){

                DECLARE_ALIGNED(8, int16_t, bS)[4];

                int qp;

                if (IS_INTRA(mb_type | h->cur_pic.mb_type[mbn_xy])) {

                    AV_WN64A(bS, 0x0003000300030003ULL);

                } else {

                    if (!CABAC(h) && IS_8x8DCT(h->cur_pic.mb_type[mbn_xy])) {

                        bS[0]= 1+((h->cbp_table[mbn_xy] & 0x4000) || sl->non_zero_count_cache[scan8[0]+0]);

                        bS[1]= 1+((h->cbp_table[mbn_xy] & 0x4000) || sl->non_zero_count_cache[scan8[0]+1]);

                        bS[2]= 1+((h->cbp_table[mbn_xy] & 0x8000) || sl->non_zero_count_cache[scan8[0]+2]);

                        bS[3]= 1+((h->cbp_table[mbn_xy] & 0x8000) || sl->non_zero_count_cache[scan8[0]+3]);

                    }else{

                    const uint8_t *mbn_nnz = h->non_zero_count[mbn_xy] + 3*4;

                    int i;

                    for( i = 0; i < 4; i++ ) {

                        bS[i] = 1 + !!(sl->non_zero_count_cache[scan8[0]+i] | mbn_nnz[i]);

                    }

                    }

                }

                // Do not use s->qscale as luma quantizer because it has not the same

                // value in IPCM macroblocks.

                qp = (h->cur_pic.qscale_table[mb_xy] + h->cur_pic.qscale_table[mbn_xy] + 1) >> 1;

                ff_tlog(h->avctx, "filter mb:%d/%d dir:%d edge:%d, QPy:%d ls:%d uvls:%d", mb_x, mb_y, dir, edge, qp, tmp_linesize, tmp_uvlinesize);

                { int i; for (i = 0; i < 4; i++) ff_tlog(h->avctx, " bS[%d]:%d", i, bS[i]); ff_tlog(h->avctx, "\n"); }

                filter_mb_edgeh( &img_y[j*linesize], tmp_linesize, bS, qp, a, b, h, 0 );

                chroma_qp_avg[0] = (sl->chroma_qp[0] + get_chroma_qp(h->ps.pps, 0, h->cur_pic.qscale_table[mbn_xy]) + 1) >> 1;

                chroma_qp_avg[1] = (sl->chroma_qp[1] + get_chroma_qp(h->ps.pps, 1, h->cur_pic.qscale_table[mbn_xy]) + 1) >> 1;

                if (chroma) {

                    if (chroma444) {

                        filter_mb_edgeh (&img_cb[j*uvlinesize], tmp_uvlinesize, bS, chroma_qp_avg[0], a, b, h, 0);

                        filter_mb_edgeh (&img_cr[j*uvlinesize], tmp_uvlinesize, bS, chroma_qp_avg[1], a, b, h, 0);

                    } else {

                        filter_mb_edgech(&img_cb[j*uvlinesize], tmp_uvlinesize, bS, chroma_qp_avg[0], a, b, h, 0);

                        filter_mb_edgech(&img_cr[j*uvlinesize], tmp_uvlinesize, bS, chroma_qp_avg[1], a, b, h, 0);

                    }

                }

            }

        }else{

            DECLARE_ALIGNED(8, int16_t, bS)[4];

            int qp;



            if( IS_INTRA(mb_type|mbm_type)) {

                AV_WN64A(bS, 0x0003000300030003ULL);

                if (   (!IS_INTERLACED(mb_type|mbm_type))

                    || ((FRAME_MBAFF(h) || (h->picture_structure != PICT_FRAME)) && (dir == 0))

                )

                    AV_WN64A(bS, 0x0004000400040004ULL);

            } else {

                int i;

                int mv_done;



                if( dir && FRAME_MBAFF(h) && IS_INTERLACED(mb_type ^ mbm_type)) {

                    AV_WN64A(bS, 0x0001000100010001ULL);

                    mv_done = 1;

                }

                else if( mask_par0 && ((mbm_type & (MB_TYPE_16x16 | (MB_TYPE_8x16 >> dir)))) ) {

                    int b_idx= 8 + 4;

                    int bn_idx= b_idx - (dir ? 8:1);



                    bS[0] = bS[1] = bS[2] = bS[3] = check_mv(sl, 8 + 4, bn_idx, mvy_limit);

                    mv_done = 1;

                }

                else

                    mv_done = 0;



                for( i = 0; i < 4; i++ ) {

                    int x = dir == 0 ? 0 : i;

                    int y = dir == 0 ? i    : 0;

                    int b_idx= 8 + 4 + x + 8*y;

                    int bn_idx= b_idx - (dir ? 8:1);



                    if (sl->non_zero_count_cache[b_idx] |

                        sl->non_zero_count_cache[bn_idx]) {

                        bS[i] = 2;

                    }

                    else if(!mv_done)

                    {

                        bS[i] = check_mv(sl, b_idx, bn_idx, mvy_limit);

                    }

                }

            }



            /* Filter edge */

            // Do not use s->qscale as luma quantizer because it has not the same

            // value in IPCM macroblocks.

            if(bS[0]+bS[1]+bS[2]+bS[3]){

                qp = (h->cur_pic.qscale_table[mb_xy] + h->cur_pic.qscale_table[mbm_xy] + 1) >> 1;

                ff_tlog(h->avctx, "filter mb:%d/%d dir:%d edge:%d, QPy:%d ls:%d uvls:%d", mb_x, mb_y, dir, edge, qp, linesize, uvlinesize);

                chroma_qp_avg[0] = (sl->chroma_qp[0] + get_chroma_qp(h->ps.pps, 0, h->cur_pic.qscale_table[mbm_xy]) + 1) >> 1;

                chroma_qp_avg[1] = (sl->chroma_qp[1] + get_chroma_qp(h->ps.pps, 1, h->cur_pic.qscale_table[mbm_xy]) + 1) >> 1;

                if( dir == 0 ) {

                    filter_mb_edgev( &img_y[0], linesize, bS, qp, a, b, h, 1 );

                    if (chroma) {

                        if (chroma444) {

                            filter_mb_edgev ( &img_cb[0], uvlinesize, bS, chroma_qp_avg[0], a, b, h, 1);

                            filter_mb_edgev ( &img_cr[0], uvlinesize, bS, chroma_qp_avg[1], a, b, h, 1);

                        } else {

                            filter_mb_edgecv( &img_cb[0], uvlinesize, bS, chroma_qp_avg[0], a, b, h, 1);

                            filter_mb_edgecv( &img_cr[0], uvlinesize, bS, chroma_qp_avg[1], a, b, h, 1);

                        }

                    }

                } else {

                    filter_mb_edgeh( &img_y[0], linesize, bS, qp, a, b, h, 1 );

                    if (chroma) {

                        if (chroma444) {

                            filter_mb_edgeh ( &img_cb[0], uvlinesize, bS, chroma_qp_avg[0], a, b, h, 1);

                            filter_mb_edgeh ( &img_cr[0], uvlinesize, bS, chroma_qp_avg[1], a, b, h, 1);

                        } else {

                            filter_mb_edgech( &img_cb[0], uvlinesize, bS, chroma_qp_avg[0], a, b, h, 1);

                            filter_mb_edgech( &img_cr[0], uvlinesize, bS, chroma_qp_avg[1], a, b, h, 1);

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

        const int deblock_edge = !IS_8x8DCT(mb_type & (edge<<24)); // (edge&1) && IS_8x8DCT(mb_type)



        if (!deblock_edge && (!chroma422 || dir == 0))

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



                bS[0] = bS[1] = bS[2] = bS[3] = check_mv(sl, b_idx, bn_idx, mvy_limit);

                mv_done = 1;

            }

            else

                mv_done = 0;



            for( i = 0; i < 4; i++ ) {

                int x = dir == 0 ? edge : i;

                int y = dir == 0 ? i    : edge;

                int b_idx= 8 + 4 + x + 8*y;

                int bn_idx= b_idx - (dir ? 8:1);



                if (sl->non_zero_count_cache[b_idx] |

                    sl->non_zero_count_cache[bn_idx]) {

                    bS[i] = 2;

                }

                else if(!mv_done)

                {

                    bS[i] = check_mv(sl, b_idx, bn_idx, mvy_limit);

                }

            }



            if(bS[0]+bS[1]+bS[2]+bS[3] == 0)

                continue;

        }



        /* Filter edge */

        // Do not use s->qscale as luma quantizer because it has not the same

        // value in IPCM macroblocks.

        qp = h->cur_pic.qscale_table[mb_xy];

        ff_tlog(h->avctx, "filter mb:%d/%d dir:%d edge:%d, QPy:%d ls:%d uvls:%d", mb_x, mb_y, dir, edge, qp, linesize, uvlinesize);

        if( dir == 0 ) {

            filter_mb_edgev( &img_y[4*edge << h->pixel_shift], linesize, bS, qp, a, b, h, 0 );

            if (chroma) {

                if (chroma444) {

                    filter_mb_edgev ( &img_cb[4*edge << h->pixel_shift], uvlinesize, bS, sl->chroma_qp[0], a, b, h, 0);

                    filter_mb_edgev ( &img_cr[4*edge << h->pixel_shift], uvlinesize, bS, sl->chroma_qp[1], a, b, h, 0);

                } else if( (edge&1) == 0 ) {

                    filter_mb_edgecv( &img_cb[2*edge << h->pixel_shift], uvlinesize, bS, sl->chroma_qp[0], a, b, h, 0);

                    filter_mb_edgecv( &img_cr[2*edge << h->pixel_shift], uvlinesize, bS, sl->chroma_qp[1], a, b, h, 0);

                }

            }

        } else {

            if (chroma422) {

                if (deblock_edge)

                    filter_mb_edgeh(&img_y[4*edge*linesize], linesize, bS, qp, a, b, h, 0);

                if (chroma) {

                    filter_mb_edgech(&img_cb[4*edge*uvlinesize], uvlinesize, bS, sl->chroma_qp[0], a, b, h, 0);

                    filter_mb_edgech(&img_cr[4*edge*uvlinesize], uvlinesize, bS, sl->chroma_qp[1], a, b, h, 0);

                }

            } else {

                filter_mb_edgeh(&img_y[4*edge*linesize], linesize, bS, qp, a, b, h, 0);

                if (chroma) {

                    if (chroma444) {

                        filter_mb_edgeh (&img_cb[4*edge*uvlinesize], uvlinesize, bS, sl->chroma_qp[0], a, b, h, 0);

                        filter_mb_edgeh (&img_cr[4*edge*uvlinesize], uvlinesize, bS, sl->chroma_qp[1], a, b, h, 0);

                    } else if ((edge&1) == 0) {

                        filter_mb_edgech(&img_cb[2*edge*uvlinesize], uvlinesize, bS, sl->chroma_qp[0], a, b, h, 0);

                        filter_mb_edgech(&img_cr[2*edge*uvlinesize], uvlinesize, bS, sl->chroma_qp[1], a, b, h, 0);

                    }

                }

            }

        }

    }

}
