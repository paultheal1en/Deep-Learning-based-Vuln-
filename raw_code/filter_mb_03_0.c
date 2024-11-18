static void filter_mb( H264Context *h, int mb_x, int mb_y ) {

    MpegEncContext * const s = &h->s;

    const int mb_xy= mb_x + mb_y*s->mb_stride;

    uint8_t *img_y  = s->current_picture.data[0] + (mb_y * 16* s->linesize  ) + mb_x * 16;

    uint8_t *img_cb = s->current_picture.data[1] + (mb_y * 8 * s->uvlinesize) + mb_x * 8;

    uint8_t *img_cr = s->current_picture.data[2] + (mb_y * 8 * s->uvlinesize) + mb_x * 8;

    int linesize, uvlinesize;

    int dir;

#if 0

    /* FIXME what's that ? */

    if( !s->decode )

        return;

#endif



    /* FIXME Implement deblocking filter for field MB */

    if( h->sps.mb_aff ) {

        return;

    }

    linesize = s->linesize;

    uvlinesize = s->uvlinesize;



    /* dir : 0 -> vertical edge, 1 -> horizontal edge */

    for( dir = 0; dir < 2; dir++ )

    {

        int start = 0;

        int edge;



        /* test picture boundary */

        if( ( dir == 0 && mb_x == 0 ) || ( dir == 1 && mb_y == 0 ) ) {

            start = 1;

        }

        /* FIXME test slice boundary */

        if( h->disable_deblocking_filter_idc == 2 ) {

        }



        /* Calculate bS */

        for( edge = start; edge < 4; edge++ ) {

            /* mbn_xy: neighbour macroblock (how that works for field ?) */

            int mbn_xy = edge > 0 ? mb_xy : ( dir == 0 ? mb_xy -1 : mb_xy - s->mb_stride );

            int bS[4];

            int qp;



            if( IS_INTRA( s->current_picture.mb_type[mb_xy] ) ||

                IS_INTRA( s->current_picture.mb_type[mbn_xy] ) ) {

                bS[0] = bS[1] = bS[2] = bS[3] = ( edge == 0 ? 4 : 3 );

            } else {

                int i;

                for( i = 0; i < 4; i++ ) {

                    static const uint8_t block_idx_xy[4][4] = {

                        { 0, 2, 8,  10}, { 1, 3, 9,  11},

                        { 4, 6, 12, 14}, { 5, 7, 13, 15}

                    };



                    int x = dir == 0 ? edge : i;

                    int y = dir == 0 ? i    : edge;

                    int xn = (x - (dir == 0 ? 1 : 0 ))&0x03;

                    int yn = (y - (dir == 0 ? 0 : 1 ))&0x03;



                    if( h->non_zero_count[mb_xy][block_idx_xy[x][y]] != 0 ||

                        h->non_zero_count[mbn_xy][block_idx_xy[xn][yn]] != 0 ) {

                        bS[i] = 2;

                    }

                    else if( h->slice_type == P_TYPE ) {

                        const int b8_xy = h->mb2b8_xy[mb_xy]+(y>>1)*h->b8_stride+(x>>1);

                        const int b8n_xy= h->mb2b8_xy[mbn_xy]+(yn>>1)*h->b8_stride+(xn>>1);

                        const int b_xy  = h->mb2b_xy[mb_xy]+y*h->b_stride+x;

                        const int bn_xy = h->mb2b_xy[mbn_xy]+yn*h->b_stride+xn;

                        if( s->current_picture.ref_index[0][b8_xy] != s->current_picture.ref_index[0][b8n_xy] ||

                            ABS( s->current_picture.motion_val[0][b_xy][0] - s->current_picture.motion_val[0][bn_xy][0] ) >= 4 ||

                            ABS( s->current_picture.motion_val[0][b_xy][1] - s->current_picture.motion_val[0][bn_xy][1] ) >= 4 )

                            bS[i] = 1;

                        else

                            bS[i] = 0;

                    }

                    else {

                        /* FIXME Add support for B frame */

                        return;

                    }

                }

            }



            /* Filter edge */

            qp = ( s->current_picture.qscale_table[mb_xy] + s->current_picture.qscale_table[mbn_xy] + 1 ) >> 1;

            if( dir == 0 ) {

                filter_mb_edgev( h, &img_y[4*edge], linesize, bS, qp );

                if( (edge&1) == 0 ) {

                    int chroma_qp = ( get_chroma_qp( h, s->current_picture.qscale_table[mb_xy] ) +

                                      get_chroma_qp( h, s->current_picture.qscale_table[mbn_xy] ) + 1 ) >> 1;

                    filter_mb_edgecv( h, &img_cb[2*edge], uvlinesize, bS, chroma_qp );

                    filter_mb_edgecv( h, &img_cr[2*edge], uvlinesize, bS, chroma_qp );

                }

            } else {

                filter_mb_edgeh( h, &img_y[4*edge*linesize], linesize, bS, qp );

                if( (edge&1) == 0 ) {

                    int chroma_qp = ( get_chroma_qp( h, s->current_picture.qscale_table[mb_xy] ) +

                                      get_chroma_qp( h, s->current_picture.qscale_table[mbn_xy] ) + 1 ) >> 1;

                    filter_mb_edgech( h, &img_cb[2*edge*uvlinesize], uvlinesize, bS, chroma_qp );

                    filter_mb_edgech( h, &img_cr[2*edge*uvlinesize], uvlinesize, bS, chroma_qp );

                }

            }

        }

    }

}
