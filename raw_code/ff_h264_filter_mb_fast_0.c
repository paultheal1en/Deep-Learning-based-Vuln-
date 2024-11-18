void ff_h264_filter_mb_fast( H264Context *h, int mb_x, int mb_y, uint8_t *img_y, uint8_t *img_cb, uint8_t *img_cr, unsigned int linesize, unsigned int uvlinesize) {

    MpegEncContext * const s = &h->s;

    int mb_xy;

    int mb_type, left_type, top_type;

    int qp, qp0, qp1, qpc, qpc0, qpc1, qp_thresh;

    int chroma = !(CONFIG_GRAY && (s->flags&CODEC_FLAG_GRAY));

    int chroma444 = CHROMA444;



    mb_xy = h->mb_xy;



    if(!h->h264dsp.h264_loop_filter_strength || h->pps.chroma_qp_diff) {

        ff_h264_filter_mb(h, mb_x, mb_y, img_y, img_cb, img_cr, linesize, uvlinesize);

        return;

    }

    assert(!FRAME_MBAFF);

    left_type= h->left_type[LTOP];

    top_type= h->top_type;



    mb_type = s->current_picture.mb_type[mb_xy];

    qp = s->current_picture.qscale_table[mb_xy];

    qp0 = s->current_picture.qscale_table[mb_xy-1];

    qp1 = s->current_picture.qscale_table[h->top_mb_xy];

    qpc = get_chroma_qp( h, 0, qp );

    qpc0 = get_chroma_qp( h, 0, qp0 );

    qpc1 = get_chroma_qp( h, 0, qp1 );

    qp0 = (qp + qp0 + 1) >> 1;

    qp1 = (qp + qp1 + 1) >> 1;

    qpc0 = (qpc + qpc0 + 1) >> 1;

    qpc1 = (qpc + qpc1 + 1) >> 1;

    qp_thresh = 15+52 - h->slice_alpha_c0_offset;

    if(qp <= qp_thresh && qp0 <= qp_thresh && qp1 <= qp_thresh &&

       qpc <= qp_thresh && qpc0 <= qp_thresh && qpc1 <= qp_thresh)

        return;



    if( IS_INTRA(mb_type) ) {

        int16_t bS4[4] = {4,4,4,4};

        int16_t bS3[4] = {3,3,3,3};

        int16_t *bSH = FIELD_PICTURE ? bS3 : bS4;

        if(left_type)

            filter_mb_edgev( &img_y[4*0], linesize, bS4, qp0, h);

        if( IS_8x8DCT(mb_type) ) {

            filter_mb_edgev( &img_y[4*2], linesize, bS3, qp, h);

            if(top_type){

                filter_mb_edgeh( &img_y[4*0*linesize], linesize, bSH, qp1, h);

            }

            filter_mb_edgeh( &img_y[4*2*linesize], linesize, bS3, qp, h);

        } else {

            filter_mb_edgev( &img_y[4*1], linesize, bS3, qp, h);

            filter_mb_edgev( &img_y[4*2], linesize, bS3, qp, h);

            filter_mb_edgev( &img_y[4*3], linesize, bS3, qp, h);

            if(top_type){

                filter_mb_edgeh( &img_y[4*0*linesize], linesize, bSH, qp1, h);

            }

            filter_mb_edgeh( &img_y[4*1*linesize], linesize, bS3, qp, h);

            filter_mb_edgeh( &img_y[4*2*linesize], linesize, bS3, qp, h);

            filter_mb_edgeh( &img_y[4*3*linesize], linesize, bS3, qp, h);

        }

        if(chroma){

            if(chroma444){

                if(left_type){

                    filter_mb_edgev( &img_cb[4*0], linesize, bS4, qpc0, h);

                    filter_mb_edgev( &img_cr[4*0], linesize, bS4, qpc0, h);

                }

                if( IS_8x8DCT(mb_type) ) {

                    filter_mb_edgev( &img_cb[4*2], linesize, bS3, qpc, h);

                    filter_mb_edgev( &img_cr[4*2], linesize, bS3, qpc, h);

                    if(top_type){

                        filter_mb_edgeh( &img_cb[4*0*linesize], linesize, bSH, qpc1, h);

                        filter_mb_edgeh( &img_cr[4*0*linesize], linesize, bSH, qpc1, h);

                    }

                    filter_mb_edgeh( &img_cb[4*2*linesize], linesize, bS3, qpc, h);

                    filter_mb_edgeh( &img_cr[4*2*linesize], linesize, bS3, qpc, h);

                } else {

                    filter_mb_edgev( &img_cb[4*1], linesize, bS3, qpc, h);

                    filter_mb_edgev( &img_cr[4*1], linesize, bS3, qpc, h);

                    filter_mb_edgev( &img_cb[4*2], linesize, bS3, qpc, h);

                    filter_mb_edgev( &img_cr[4*2], linesize, bS3, qpc, h);

                    filter_mb_edgev( &img_cb[4*3], linesize, bS3, qpc, h);

                    filter_mb_edgev( &img_cr[4*3], linesize, bS3, qpc, h);

                    if(top_type){

                        filter_mb_edgeh( &img_cb[4*0*linesize], linesize, bSH, qpc1, h);

                        filter_mb_edgeh( &img_cr[4*0*linesize], linesize, bSH, qpc1, h);

                    }

                    filter_mb_edgeh( &img_cb[4*1*linesize], linesize, bS3, qpc, h);

                    filter_mb_edgeh( &img_cr[4*1*linesize], linesize, bS3, qpc, h);

                    filter_mb_edgeh( &img_cb[4*2*linesize], linesize, bS3, qpc, h);

                    filter_mb_edgeh( &img_cr[4*2*linesize], linesize, bS3, qpc, h);

                    filter_mb_edgeh( &img_cb[4*3*linesize], linesize, bS3, qpc, h);

                    filter_mb_edgeh( &img_cr[4*3*linesize], linesize, bS3, qpc, h);

                }

            }else{

                if(left_type){

                    filter_mb_edgecv( &img_cb[2*0], uvlinesize, bS4, qpc0, h);

                    filter_mb_edgecv( &img_cr[2*0], uvlinesize, bS4, qpc0, h);

                }

                filter_mb_edgecv( &img_cb[2*2], uvlinesize, bS3, qpc, h);

                filter_mb_edgecv( &img_cr[2*2], uvlinesize, bS3, qpc, h);

                if(top_type){

                    filter_mb_edgech( &img_cb[2*0*uvlinesize], uvlinesize, bSH, qpc1, h);

                    filter_mb_edgech( &img_cr[2*0*uvlinesize], uvlinesize, bSH, qpc1, h);

                }

                filter_mb_edgech( &img_cb[2*2*uvlinesize], uvlinesize, bS3, qpc, h);

                filter_mb_edgech( &img_cr[2*2*uvlinesize], uvlinesize, bS3, qpc, h);

            }

        }

        return;

    } else {

        LOCAL_ALIGNED_8(int16_t, bS, [2], [4][4]);

        int edges;

        if( IS_8x8DCT(mb_type) && (h->cbp&7) == 7 ) {

            edges = 4;

            AV_WN64A(bS[0][0], 0x0002000200020002ULL);

            AV_WN64A(bS[0][2], 0x0002000200020002ULL);

            AV_WN64A(bS[1][0], 0x0002000200020002ULL);

            AV_WN64A(bS[1][2], 0x0002000200020002ULL);

        } else {

            int mask_edge1 = (3*(((5*mb_type)>>5)&1)) | (mb_type>>4); //(mb_type & (MB_TYPE_16x16 | MB_TYPE_8x16)) ? 3 : (mb_type & MB_TYPE_16x8) ? 1 : 0;

            int mask_edge0 = 3*((mask_edge1>>1) & ((5*left_type)>>5)&1); // (mb_type & (MB_TYPE_16x16 | MB_TYPE_8x16)) && (h->left_type[LTOP] & (MB_TYPE_16x16 | MB_TYPE_8x16)) ? 3 : 0;

            int step =  1+(mb_type>>24); //IS_8x8DCT(mb_type) ? 2 : 1;

            edges = 4 - 3*((mb_type>>3) & !(h->cbp & 15)); //(mb_type & MB_TYPE_16x16) && !(h->cbp & 15) ? 1 : 4;

            h->h264dsp.h264_loop_filter_strength( bS, h->non_zero_count_cache, h->ref_cache, h->mv_cache,

                                              h->list_count==2, edges, step, mask_edge0, mask_edge1, FIELD_PICTURE);

        }

        if( IS_INTRA(left_type) )

            AV_WN64A(bS[0][0], 0x0004000400040004ULL);

        if( IS_INTRA(top_type) )

            AV_WN64A(bS[1][0], FIELD_PICTURE ? 0x0003000300030003ULL : 0x0004000400040004ULL);



#define FILTER(hv,dir,edge)\

        if(AV_RN64A(bS[dir][edge])) {                                   \

            filter_mb_edge##hv( &img_y[4*edge*(dir?linesize:1)], linesize, bS[dir][edge], edge ? qp : qp##dir, h );\

            if(chroma){\

                if(chroma444){\

                    filter_mb_edge##hv( &img_cb[4*edge*(dir?linesize:1)], linesize, bS[dir][edge], edge ? qpc : qpc##dir, h );\

                    filter_mb_edge##hv( &img_cr[4*edge*(dir?linesize:1)], linesize, bS[dir][edge], edge ? qpc : qpc##dir, h );\

                } else if(!(edge&1)) {\

                    filter_mb_edgec##hv( &img_cb[2*edge*(dir?uvlinesize:1)], uvlinesize, bS[dir][edge], edge ? qpc : qpc##dir, h );\

                    filter_mb_edgec##hv( &img_cr[2*edge*(dir?uvlinesize:1)], uvlinesize, bS[dir][edge], edge ? qpc : qpc##dir, h );\

                }\

            }\

        }

        if(left_type)

            FILTER(v,0,0);

        if( edges == 1 ) {

            if(top_type)

                FILTER(h,1,0);

        } else if( IS_8x8DCT(mb_type) ) {

            FILTER(v,0,2);

            if(top_type)

                FILTER(h,1,0);

            FILTER(h,1,2);

        } else {

            FILTER(v,0,1);

            FILTER(v,0,2);

            FILTER(v,0,3);

            if(top_type)

                FILTER(h,1,0);

            FILTER(h,1,1);

            FILTER(h,1,2);

            FILTER(h,1,3);

        }

#undef FILTER

    }

}
