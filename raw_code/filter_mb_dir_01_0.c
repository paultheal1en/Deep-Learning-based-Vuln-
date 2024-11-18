static av_always_inline void filter_mb_dir(H264Context *h, int mb_x, int mb_y, uint8_t *img_y, uint8_t *img_cb, uint8_t *img_cr, unsigned int linesize, unsigned int uvlinesize, int mb_xy, int mb_type, int mvy_limit, int first_vertical_edge_done, int dir) {

    MpegEncContext * const s = &h->s;

    int edge;

    const int mbm_xy = dir == 0 ? mb_xy -1 : h->top_mb_xy;

    const int mbm_type = dir == 0 ? h->left_type[0] : h->top_type;



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

                DECLARE_ALIGNED_8(int16_t, bS)[4];

                int qp;

                if( IS_INTRA(mb_type|s->current_picture.mb_type[mbn_xy]) ) {

                    *(uint64_t*)bS= 0x0003000300030003ULL;

                } else {

                    const uint8_t *mbn_nnz = h->non_zero_count[mbn_xy] + 4+3*8; //FIXME  8x8dct?

                    int i;

                    for( i = 0; i < 4; i++ ) {

                        bS[i] = 1 + !!(h->non_zero_count_cache[scan8[0]+i] | mbn_nnz[i]);

                    }

                }

                // Do not use s->qscale as luma quantizer because it has not the same

                // value in IPCM macroblocks.

                qp = ( s->current_picture.qscale_table[mb_xy] + s->current_picture.qscale_table[mbn_xy] + 1 ) >> 1;

                tprintf(s->avctx, "filter mb:%d/%d dir:%d edge:%d, QPy:%d ls:%d uvls:%d", mb_x, mb_y, dir, edge, qp, tmp_linesize, tmp_uvlinesize);

                { int i; for (i = 0; i < 4; i++) tprintf(s->avctx, " bS[%d]:%d", i, bS[i]); tprintf(s->avctx, "\n"); }

                filter_mb_edgeh( &img_y[j*linesize], tmp_linesize, bS, qp, h );

                filter_mb_edgech( &img_cb[j*uvlinesize], tmp_uvlinesize, bS,

                                ( h->chroma_qp[0] + get_chroma_qp( h, 0, s->current_picture.qscale_table[mbn_xy] ) + 1 ) >> 1, h);

                filter_mb_edgech( &img_cr[j*uvlinesize], tmp_uvlinesize, bS,

                                ( h->chroma_qp[1] + get_chroma_qp( h, 1, s->current_picture.qscale_table[mbn_xy] ) + 1 ) >> 1, h);

            }

        }else{

            DECLARE_ALIGNED_8(int16_t, bS)[4];

            int qp;



            if( IS_INTRA(mb_type|mbm_type)) {

                *(uint64_t*)bS= 0x0003000300030003ULL;

                if (   (!IS_INTERLACED(mb_type|mbm_type))

                    || ((FRAME_MBAFF || (s->picture_structure != PICT_FRAME)) && (dir == 0))

                )

                    *(uint64_t*)bS= 0x0004000400040004ULL;

            } else {

                int i, l;

                int mv_done;



                if( dir && FRAME_MBAFF && IS_INTERLACED(mb_type ^ mbm_type)) {

                    *(uint64_t*)bS= 0x0001000100010001ULL;

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

                qp = ( s->current_picture.qscale_table[mb_xy] + s->current_picture.qscale_table[mbm_xy] + 1 ) >> 1;

                //tprintf(s->avctx, "filter mb:%d/%d dir:%d edge:%d, QPy:%d, QPc:%d, QPcn:%d\n", mb_x, mb_y, dir, edge, qp, h->chroma_qp[0], s->current_picture.qscale_table[mbn_xy]);

                tprintf(s->avctx, "filter mb:%d/%d dir:%d edge:%d, QPy:%d ls:%d uvls:%d", mb_x, mb_y, dir, edge, qp, linesize, uvlinesize);

                //{ int i; for (i = 0; i < 4; i++) tprintf(s->avctx, " bS[%d]:%d", i, bS[i]); tprintf(s->avctx, "\n"); }

                if( dir == 0 ) {

                    filter_mb_edgev( &img_y[0], linesize, bS, qp, h );

                    {

                        int qp= ( h->chroma_qp[0] + get_chroma_qp( h, 0, s->current_picture.qscale_table[mbm_xy] ) + 1 ) >> 1;

                        filter_mb_edgecv( &img_cb[0], uvlinesize, bS, qp, h);

                        if(h->pps.chroma_qp_diff)

                            qp= ( h->chroma_qp[1] + get_chroma_qp( h, 1, s->current_picture.qscale_table[mbm_xy] ) + 1 ) >> 1;

                        filter_mb_edgecv( &img_cr[0], uvlinesize, bS, qp, h);

                    }

                } else {

                    filter_mb_edgeh( &img_y[0], linesize, bS, qp, h );

                    {

                        int qp= ( h->chroma_qp[0] + get_chroma_qp( h, 0, s->current_picture.qscale_table[mbm_xy] ) + 1 ) >> 1;

                        filter_mb_edgech( &img_cb[0], uvlinesize, bS, qp, h);

                        if(h->pps.chroma_qp_diff)

                            qp= ( h->chroma_qp[1] + get_chroma_qp( h, 1, s->current_picture.qscale_table[mbm_xy] ) + 1 ) >> 1;

                        filter_mb_edgech( &img_cr[0], uvlinesize, bS, qp, h);

                    }

                }

            }

        }

    }



    /* Calculate bS */

    for( edge = 1; edge < edges; edge++ ) {

        DECLARE_ALIGNED_8(int16_t, bS)[4];

        int qp;



        if( IS_8x8DCT(mb_type & (edge<<24)) ) // (edge&1) && IS_8x8DCT(mb_type)

            continue;



        if( IS_INTRA(mb_type)) {

            *(uint64_t*)bS= 0x0003000300030003ULL;

        } else {

            int i, l;

            int mv_done;



            if( edge & mask_edge ) {

                *(uint64_t*)bS= 0;

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

        qp = s->current_picture.qscale_table[mb_xy];

        //tprintf(s->avctx, "filter mb:%d/%d dir:%d edge:%d, QPy:%d, QPc:%d, QPcn:%d\n", mb_x, mb_y, dir, edge, qp, h->chroma_qp[0], s->current_picture.qscale_table[mbn_xy]);

        tprintf(s->avctx, "filter mb:%d/%d dir:%d edge:%d, QPy:%d ls:%d uvls:%d", mb_x, mb_y, dir, edge, qp, linesize, uvlinesize);

        //{ int i; for (i = 0; i < 4; i++) tprintf(s->avctx, " bS[%d]:%d", i, bS[i]); tprintf(s->avctx, "\n"); }

        if( dir == 0 ) {

            filter_mb_edgev( &img_y[4*edge], linesize, bS, qp, h );

            if( (edge&1) == 0 ) {

                filter_mb_edgecv( &img_cb[2*edge], uvlinesize, bS, h->chroma_qp[0], h);

                filter_mb_edgecv( &img_cr[2*edge], uvlinesize, bS, h->chroma_qp[1], h);

            }

        } else {

            filter_mb_edgeh( &img_y[4*edge*linesize], linesize, bS, qp, h );

            if( (edge&1) == 0 ) {

                filter_mb_edgech( &img_cb[2*edge*uvlinesize], uvlinesize, bS, h->chroma_qp[0], h);

                filter_mb_edgech( &img_cr[2*edge*uvlinesize], uvlinesize, bS, h->chroma_qp[1], h);

            }

        }

    }

}
