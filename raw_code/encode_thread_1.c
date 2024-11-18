static int encode_thread(AVCodecContext *c, void *arg){

    MpegEncContext *s= *(void**)arg;

    int mb_x, mb_y, pdif = 0;

    int chr_h= 16>>s->chroma_y_shift;

    int i, j;

    MpegEncContext best_s, backup_s;

    uint8_t bit_buf[2][MAX_MB_BYTES];

    uint8_t bit_buf2[2][MAX_MB_BYTES];

    uint8_t bit_buf_tex[2][MAX_MB_BYTES];

    PutBitContext pb[2], pb2[2], tex_pb[2];



    for(i=0; i<2; i++){

        init_put_bits(&pb    [i], bit_buf    [i], MAX_MB_BYTES);

        init_put_bits(&pb2   [i], bit_buf2   [i], MAX_MB_BYTES);

        init_put_bits(&tex_pb[i], bit_buf_tex[i], MAX_MB_BYTES);

    }



    s->last_bits= put_bits_count(&s->pb);

    s->mv_bits=0;

    s->misc_bits=0;

    s->i_tex_bits=0;

    s->p_tex_bits=0;

    s->i_count=0;

    s->f_count=0;

    s->b_count=0;

    s->skip_count=0;



    for(i=0; i<3; i++){

        /* init last dc values */

        /* note: quant matrix value (8) is implied here */

        s->last_dc[i] = 128 << s->intra_dc_precision;



        s->current_picture.f.error[i] = 0;

    }

    s->mb_skip_run = 0;

    memset(s->last_mv, 0, sizeof(s->last_mv));



    s->last_mv_dir = 0;



    switch(s->codec_id){

    case AV_CODEC_ID_H263:

    case AV_CODEC_ID_H263P:

    case AV_CODEC_ID_FLV1:

        if (CONFIG_H263_ENCODER)

            s->gob_index = ff_h263_get_gob_height(s);

        break;

    case AV_CODEC_ID_MPEG4:

        if(CONFIG_MPEG4_ENCODER && s->partitioned_frame)

            ff_mpeg4_init_partitions(s);

        break;

    }



    s->resync_mb_x=0;

    s->resync_mb_y=0;

    s->first_slice_line = 1;

    s->ptr_lastgob = s->pb.buf;

    for(mb_y= s->start_mb_y; mb_y < s->end_mb_y; mb_y++) {

        s->mb_x=0;

        s->mb_y= mb_y;



        ff_set_qscale(s, s->qscale);

        ff_init_block_index(s);



        for(mb_x=0; mb_x < s->mb_width; mb_x++) {

            int xy= mb_y*s->mb_stride + mb_x; // removed const, H261 needs to adjust this

            int mb_type= s->mb_type[xy];

//            int d;

            int dmin= INT_MAX;

            int dir;



            if(s->pb.buf_end - s->pb.buf - (put_bits_count(&s->pb)>>3) < MAX_MB_BYTES){

                av_log(s->avctx, AV_LOG_ERROR, "encoded frame too large\n");

                return -1;

            }

            if(s->data_partitioning){

                if(   s->pb2   .buf_end - s->pb2   .buf - (put_bits_count(&s->    pb2)>>3) < MAX_MB_BYTES

                   || s->tex_pb.buf_end - s->tex_pb.buf - (put_bits_count(&s->tex_pb )>>3) < MAX_MB_BYTES){

                    av_log(s->avctx, AV_LOG_ERROR, "encoded frame too large\n");

                    return -1;

                }

            }



            s->mb_x = mb_x;

            s->mb_y = mb_y;  // moved into loop, can get changed by H.261

            ff_update_block_index(s);



            if(CONFIG_H261_ENCODER && s->codec_id == AV_CODEC_ID_H261){

                ff_h261_reorder_mb_index(s);

                xy= s->mb_y*s->mb_stride + s->mb_x;

                mb_type= s->mb_type[xy];

            }



            /* write gob / video packet header  */

            if(s->rtp_mode){

                int current_packet_size, is_gob_start;



                current_packet_size= ((put_bits_count(&s->pb)+7)>>3) - (s->ptr_lastgob - s->pb.buf);



                is_gob_start= s->avctx->rtp_payload_size && current_packet_size >= s->avctx->rtp_payload_size && mb_y + mb_x>0;



                if(s->start_mb_y == mb_y && mb_y > 0 && mb_x==0) is_gob_start=1;



                switch(s->codec_id){

                case AV_CODEC_ID_H263:

                case AV_CODEC_ID_H263P:

                    if(!s->h263_slice_structured)

                        if(s->mb_x || s->mb_y%s->gob_index) is_gob_start=0;

                    break;

                case AV_CODEC_ID_MPEG2VIDEO:

                    if(s->mb_x==0 && s->mb_y!=0) is_gob_start=1;

                case AV_CODEC_ID_MPEG1VIDEO:

                    if(s->mb_skip_run) is_gob_start=0;

                    break;

                }



                if(is_gob_start){

                    if(s->start_mb_y != mb_y || mb_x!=0){

                        write_slice_end(s);



                        if(CONFIG_MPEG4_ENCODER && s->codec_id==AV_CODEC_ID_MPEG4 && s->partitioned_frame){

                            ff_mpeg4_init_partitions(s);

                        }

                    }



                    assert((put_bits_count(&s->pb)&7) == 0);

                    current_packet_size= put_bits_ptr(&s->pb) - s->ptr_lastgob;



                    if (s->error_rate && s->resync_mb_x + s->resync_mb_y > 0) {

                        int r= put_bits_count(&s->pb)/8 + s->picture_number + 16 + s->mb_x + s->mb_y;

                        int d = 100 / s->error_rate;

                        if(r % d == 0){

                            current_packet_size=0;

                            s->pb.buf_ptr= s->ptr_lastgob;

                            assert(put_bits_ptr(&s->pb) == s->ptr_lastgob);

                        }

                    }



                    if (s->avctx->rtp_callback){

                        int number_mb = (mb_y - s->resync_mb_y)*s->mb_width + mb_x - s->resync_mb_x;

                        s->avctx->rtp_callback(s->avctx, s->ptr_lastgob, current_packet_size, number_mb);

                    }

                    update_mb_info(s, 1);



                    switch(s->codec_id){

                    case AV_CODEC_ID_MPEG4:

                        if (CONFIG_MPEG4_ENCODER) {

                            ff_mpeg4_encode_video_packet_header(s);

                            ff_mpeg4_clean_buffers(s);

                        }

                    break;

                    case AV_CODEC_ID_MPEG1VIDEO:

                    case AV_CODEC_ID_MPEG2VIDEO:

                        if (CONFIG_MPEG1VIDEO_ENCODER || CONFIG_MPEG2VIDEO_ENCODER) {

                            ff_mpeg1_encode_slice_header(s);

                            ff_mpeg1_clean_buffers(s);

                        }

                    break;

                    case AV_CODEC_ID_H263:

                    case AV_CODEC_ID_H263P:

                        if (CONFIG_H263_ENCODER)

                            ff_h263_encode_gob_header(s, mb_y);

                    break;

                    }



                    if(s->flags&CODEC_FLAG_PASS1){

                        int bits= put_bits_count(&s->pb);

                        s->misc_bits+= bits - s->last_bits;

                        s->last_bits= bits;

                    }



                    s->ptr_lastgob += current_packet_size;

                    s->first_slice_line=1;

                    s->resync_mb_x=mb_x;

                    s->resync_mb_y=mb_y;

                }

            }



            if(  (s->resync_mb_x   == s->mb_x)

               && s->resync_mb_y+1 == s->mb_y){

                s->first_slice_line=0;

            }



            s->mb_skipped=0;

            s->dquant=0; //only for QP_RD



            update_mb_info(s, 0);



            if (mb_type & (mb_type-1) || (s->mpv_flags & FF_MPV_FLAG_QP_RD)) { // more than 1 MB type possible or FF_MPV_FLAG_QP_RD

                int next_block=0;

                int pb_bits_count, pb2_bits_count, tex_pb_bits_count;



                copy_context_before_encode(&backup_s, s, -1);

                backup_s.pb= s->pb;

                best_s.data_partitioning= s->data_partitioning;

                best_s.partitioned_frame= s->partitioned_frame;

                if(s->data_partitioning){

                    backup_s.pb2= s->pb2;

                    backup_s.tex_pb= s->tex_pb;

                }



                if(mb_type&CANDIDATE_MB_TYPE_INTER){

                    s->mv_dir = MV_DIR_FORWARD;

                    s->mv_type = MV_TYPE_16X16;

                    s->mb_intra= 0;

                    s->mv[0][0][0] = s->p_mv_table[xy][0];

                    s->mv[0][0][1] = s->p_mv_table[xy][1];

                    encode_mb_hq(s, &backup_s, &best_s, CANDIDATE_MB_TYPE_INTER, pb, pb2, tex_pb,

                                 &dmin, &next_block, s->mv[0][0][0], s->mv[0][0][1]);

                }

                if(mb_type&CANDIDATE_MB_TYPE_INTER_I){

                    s->mv_dir = MV_DIR_FORWARD;

                    s->mv_type = MV_TYPE_FIELD;

                    s->mb_intra= 0;

                    for(i=0; i<2; i++){

                        j= s->field_select[0][i] = s->p_field_select_table[i][xy];

                        s->mv[0][i][0] = s->p_field_mv_table[i][j][xy][0];

                        s->mv[0][i][1] = s->p_field_mv_table[i][j][xy][1];

                    }

                    encode_mb_hq(s, &backup_s, &best_s, CANDIDATE_MB_TYPE_INTER_I, pb, pb2, tex_pb,

                                 &dmin, &next_block, 0, 0);

                }

                if(mb_type&CANDIDATE_MB_TYPE_SKIPPED){

                    s->mv_dir = MV_DIR_FORWARD;

                    s->mv_type = MV_TYPE_16X16;

                    s->mb_intra= 0;

                    s->mv[0][0][0] = 0;

                    s->mv[0][0][1] = 0;

                    encode_mb_hq(s, &backup_s, &best_s, CANDIDATE_MB_TYPE_SKIPPED, pb, pb2, tex_pb,

                                 &dmin, &next_block, s->mv[0][0][0], s->mv[0][0][1]);

                }

                if(mb_type&CANDIDATE_MB_TYPE_INTER4V){

                    s->mv_dir = MV_DIR_FORWARD;

                    s->mv_type = MV_TYPE_8X8;

                    s->mb_intra= 0;

                    for(i=0; i<4; i++){

                        s->mv[0][i][0] = s->current_picture.motion_val[0][s->block_index[i]][0];

                        s->mv[0][i][1] = s->current_picture.motion_val[0][s->block_index[i]][1];

                    }

                    encode_mb_hq(s, &backup_s, &best_s, CANDIDATE_MB_TYPE_INTER4V, pb, pb2, tex_pb,

                                 &dmin, &next_block, 0, 0);

                }

                if(mb_type&CANDIDATE_MB_TYPE_FORWARD){

                    s->mv_dir = MV_DIR_FORWARD;

                    s->mv_type = MV_TYPE_16X16;

                    s->mb_intra= 0;

                    s->mv[0][0][0] = s->b_forw_mv_table[xy][0];

                    s->mv[0][0][1] = s->b_forw_mv_table[xy][1];

                    encode_mb_hq(s, &backup_s, &best_s, CANDIDATE_MB_TYPE_FORWARD, pb, pb2, tex_pb,

                                 &dmin, &next_block, s->mv[0][0][0], s->mv[0][0][1]);

                }

                if(mb_type&CANDIDATE_MB_TYPE_BACKWARD){

                    s->mv_dir = MV_DIR_BACKWARD;

                    s->mv_type = MV_TYPE_16X16;

                    s->mb_intra= 0;

                    s->mv[1][0][0] = s->b_back_mv_table[xy][0];

                    s->mv[1][0][1] = s->b_back_mv_table[xy][1];

                    encode_mb_hq(s, &backup_s, &best_s, CANDIDATE_MB_TYPE_BACKWARD, pb, pb2, tex_pb,

                                 &dmin, &next_block, s->mv[1][0][0], s->mv[1][0][1]);

                }

                if(mb_type&CANDIDATE_MB_TYPE_BIDIR){

                    s->mv_dir = MV_DIR_FORWARD | MV_DIR_BACKWARD;

                    s->mv_type = MV_TYPE_16X16;

                    s->mb_intra= 0;

                    s->mv[0][0][0] = s->b_bidir_forw_mv_table[xy][0];

                    s->mv[0][0][1] = s->b_bidir_forw_mv_table[xy][1];

                    s->mv[1][0][0] = s->b_bidir_back_mv_table[xy][0];

                    s->mv[1][0][1] = s->b_bidir_back_mv_table[xy][1];

                    encode_mb_hq(s, &backup_s, &best_s, CANDIDATE_MB_TYPE_BIDIR, pb, pb2, tex_pb,

                                 &dmin, &next_block, 0, 0);

                }

                if(mb_type&CANDIDATE_MB_TYPE_FORWARD_I){

                    s->mv_dir = MV_DIR_FORWARD;

                    s->mv_type = MV_TYPE_FIELD;

                    s->mb_intra= 0;

                    for(i=0; i<2; i++){

                        j= s->field_select[0][i] = s->b_field_select_table[0][i][xy];

                        s->mv[0][i][0] = s->b_field_mv_table[0][i][j][xy][0];

                        s->mv[0][i][1] = s->b_field_mv_table[0][i][j][xy][1];

                    }

                    encode_mb_hq(s, &backup_s, &best_s, CANDIDATE_MB_TYPE_FORWARD_I, pb, pb2, tex_pb,

                                 &dmin, &next_block, 0, 0);

                }

                if(mb_type&CANDIDATE_MB_TYPE_BACKWARD_I){

                    s->mv_dir = MV_DIR_BACKWARD;

                    s->mv_type = MV_TYPE_FIELD;

                    s->mb_intra= 0;

                    for(i=0; i<2; i++){

                        j= s->field_select[1][i] = s->b_field_select_table[1][i][xy];

                        s->mv[1][i][0] = s->b_field_mv_table[1][i][j][xy][0];

                        s->mv[1][i][1] = s->b_field_mv_table[1][i][j][xy][1];

                    }

                    encode_mb_hq(s, &backup_s, &best_s, CANDIDATE_MB_TYPE_BACKWARD_I, pb, pb2, tex_pb,

                                 &dmin, &next_block, 0, 0);

                }

                if(mb_type&CANDIDATE_MB_TYPE_BIDIR_I){

                    s->mv_dir = MV_DIR_FORWARD | MV_DIR_BACKWARD;

                    s->mv_type = MV_TYPE_FIELD;

                    s->mb_intra= 0;

                    for(dir=0; dir<2; dir++){

                        for(i=0; i<2; i++){

                            j= s->field_select[dir][i] = s->b_field_select_table[dir][i][xy];

                            s->mv[dir][i][0] = s->b_field_mv_table[dir][i][j][xy][0];

                            s->mv[dir][i][1] = s->b_field_mv_table[dir][i][j][xy][1];

                        }

                    }

                    encode_mb_hq(s, &backup_s, &best_s, CANDIDATE_MB_TYPE_BIDIR_I, pb, pb2, tex_pb,

                                 &dmin, &next_block, 0, 0);

                }

                if(mb_type&CANDIDATE_MB_TYPE_INTRA){

                    s->mv_dir = 0;

                    s->mv_type = MV_TYPE_16X16;

                    s->mb_intra= 1;

                    s->mv[0][0][0] = 0;

                    s->mv[0][0][1] = 0;

                    encode_mb_hq(s, &backup_s, &best_s, CANDIDATE_MB_TYPE_INTRA, pb, pb2, tex_pb,

                                 &dmin, &next_block, 0, 0);

                    if(s->h263_pred || s->h263_aic){

                        if(best_s.mb_intra)

                            s->mbintra_table[mb_x + mb_y*s->mb_stride]=1;

                        else

                            ff_clean_intra_table_entries(s); //old mode?

                    }

                }



                if ((s->mpv_flags & FF_MPV_FLAG_QP_RD) && dmin < INT_MAX) {

                    if(best_s.mv_type==MV_TYPE_16X16){ //FIXME move 4mv after QPRD

                        const int last_qp= backup_s.qscale;

                        int qpi, qp, dc[6];

                        int16_t ac[6][16];

                        const int mvdir= (best_s.mv_dir&MV_DIR_BACKWARD) ? 1 : 0;

                        static const int dquant_tab[4]={-1,1,-2,2};



                        assert(backup_s.dquant == 0);



                        //FIXME intra

                        s->mv_dir= best_s.mv_dir;

                        s->mv_type = MV_TYPE_16X16;

                        s->mb_intra= best_s.mb_intra;

                        s->mv[0][0][0] = best_s.mv[0][0][0];

                        s->mv[0][0][1] = best_s.mv[0][0][1];

                        s->mv[1][0][0] = best_s.mv[1][0][0];

                        s->mv[1][0][1] = best_s.mv[1][0][1];



                        qpi = s->pict_type == AV_PICTURE_TYPE_B ? 2 : 0;

                        for(; qpi<4; qpi++){

                            int dquant= dquant_tab[qpi];

                            qp= last_qp + dquant;

                            if(qp < s->avctx->qmin || qp > s->avctx->qmax)

                                continue;

                            backup_s.dquant= dquant;

                            if(s->mb_intra && s->dc_val[0]){

                                for(i=0; i<6; i++){

                                    dc[i]= s->dc_val[0][ s->block_index[i] ];

                                    memcpy(ac[i], s->ac_val[0][s->block_index[i]], sizeof(int16_t)*16);

                                }

                            }



                            encode_mb_hq(s, &backup_s, &best_s, CANDIDATE_MB_TYPE_INTER /* wrong but unused */, pb, pb2, tex_pb,

                                         &dmin, &next_block, s->mv[mvdir][0][0], s->mv[mvdir][0][1]);

                            if(best_s.qscale != qp){

                                if(s->mb_intra && s->dc_val[0]){

                                    for(i=0; i<6; i++){

                                        s->dc_val[0][ s->block_index[i] ]= dc[i];

                                        memcpy(s->ac_val[0][s->block_index[i]], ac[i], sizeof(int16_t)*16);

                                    }

                                }

                            }

                        }

                    }

                }

                if(CONFIG_MPEG4_ENCODER && mb_type&CANDIDATE_MB_TYPE_DIRECT){

                    int mx= s->b_direct_mv_table[xy][0];

                    int my= s->b_direct_mv_table[xy][1];



                    backup_s.dquant = 0;

                    s->mv_dir = MV_DIR_FORWARD | MV_DIR_BACKWARD | MV_DIRECT;

                    s->mb_intra= 0;

                    ff_mpeg4_set_direct_mv(s, mx, my);

                    encode_mb_hq(s, &backup_s, &best_s, CANDIDATE_MB_TYPE_DIRECT, pb, pb2, tex_pb,

                                 &dmin, &next_block, mx, my);

                }

                if(CONFIG_MPEG4_ENCODER && mb_type&CANDIDATE_MB_TYPE_DIRECT0){

                    backup_s.dquant = 0;

                    s->mv_dir = MV_DIR_FORWARD | MV_DIR_BACKWARD | MV_DIRECT;

                    s->mb_intra= 0;

                    ff_mpeg4_set_direct_mv(s, 0, 0);

                    encode_mb_hq(s, &backup_s, &best_s, CANDIDATE_MB_TYPE_DIRECT, pb, pb2, tex_pb,

                                 &dmin, &next_block, 0, 0);

                }

                if (!best_s.mb_intra && s->mpv_flags & FF_MPV_FLAG_SKIP_RD) {

                    int coded=0;

                    for(i=0; i<6; i++)

                        coded |= s->block_last_index[i];

                    if(coded){

                        int mx,my;

                        memcpy(s->mv, best_s.mv, sizeof(s->mv));

                        if(CONFIG_MPEG4_ENCODER && best_s.mv_dir & MV_DIRECT){

                            mx=my=0; //FIXME find the one we actually used

                            ff_mpeg4_set_direct_mv(s, mx, my);

                        }else if(best_s.mv_dir&MV_DIR_BACKWARD){

                            mx= s->mv[1][0][0];

                            my= s->mv[1][0][1];

                        }else{

                            mx= s->mv[0][0][0];

                            my= s->mv[0][0][1];

                        }



                        s->mv_dir= best_s.mv_dir;

                        s->mv_type = best_s.mv_type;

                        s->mb_intra= 0;

/*                        s->mv[0][0][0] = best_s.mv[0][0][0];

                        s->mv[0][0][1] = best_s.mv[0][0][1];

                        s->mv[1][0][0] = best_s.mv[1][0][0];

                        s->mv[1][0][1] = best_s.mv[1][0][1];*/

                        backup_s.dquant= 0;

                        s->skipdct=1;

                        encode_mb_hq(s, &backup_s, &best_s, CANDIDATE_MB_TYPE_INTER /* wrong but unused */, pb, pb2, tex_pb,

                                        &dmin, &next_block, mx, my);

                        s->skipdct=0;

                    }

                }



                s->current_picture.qscale_table[xy] = best_s.qscale;



                copy_context_after_encode(s, &best_s, -1);



                pb_bits_count= put_bits_count(&s->pb);

                flush_put_bits(&s->pb);

                avpriv_copy_bits(&backup_s.pb, bit_buf[next_block^1], pb_bits_count);

                s->pb= backup_s.pb;



                if(s->data_partitioning){

                    pb2_bits_count= put_bits_count(&s->pb2);

                    flush_put_bits(&s->pb2);

                    avpriv_copy_bits(&backup_s.pb2, bit_buf2[next_block^1], pb2_bits_count);

                    s->pb2= backup_s.pb2;



                    tex_pb_bits_count= put_bits_count(&s->tex_pb);

                    flush_put_bits(&s->tex_pb);

                    avpriv_copy_bits(&backup_s.tex_pb, bit_buf_tex[next_block^1], tex_pb_bits_count);

                    s->tex_pb= backup_s.tex_pb;

                }

                s->last_bits= put_bits_count(&s->pb);



                if (CONFIG_H263_ENCODER &&

                    s->out_format == FMT_H263 && s->pict_type!=AV_PICTURE_TYPE_B)

                    ff_h263_update_motion_val(s);



                if(next_block==0){ //FIXME 16 vs linesize16

                    s->hdsp.put_pixels_tab[0][0](s->dest[0], s->rd_scratchpad                     , s->linesize  ,16);

                    s->hdsp.put_pixels_tab[1][0](s->dest[1], s->rd_scratchpad + 16*s->linesize    , s->uvlinesize, 8);

                    s->hdsp.put_pixels_tab[1][0](s->dest[2], s->rd_scratchpad + 16*s->linesize + 8, s->uvlinesize, 8);

                }



                if(s->avctx->mb_decision == FF_MB_DECISION_BITS)

                    ff_MPV_decode_mb(s, s->block);

            } else {

                int motion_x = 0, motion_y = 0;

                s->mv_type=MV_TYPE_16X16;

                // only one MB-Type possible



                switch(mb_type){

                case CANDIDATE_MB_TYPE_INTRA:

                    s->mv_dir = 0;

                    s->mb_intra= 1;

                    motion_x= s->mv[0][0][0] = 0;

                    motion_y= s->mv[0][0][1] = 0;

                    break;

                case CANDIDATE_MB_TYPE_INTER:

                    s->mv_dir = MV_DIR_FORWARD;

                    s->mb_intra= 0;

                    motion_x= s->mv[0][0][0] = s->p_mv_table[xy][0];

                    motion_y= s->mv[0][0][1] = s->p_mv_table[xy][1];

                    break;

                case CANDIDATE_MB_TYPE_INTER_I:

                    s->mv_dir = MV_DIR_FORWARD;

                    s->mv_type = MV_TYPE_FIELD;

                    s->mb_intra= 0;

                    for(i=0; i<2; i++){

                        j= s->field_select[0][i] = s->p_field_select_table[i][xy];

                        s->mv[0][i][0] = s->p_field_mv_table[i][j][xy][0];

                        s->mv[0][i][1] = s->p_field_mv_table[i][j][xy][1];

                    }

                    break;

                case CANDIDATE_MB_TYPE_INTER4V:

                    s->mv_dir = MV_DIR_FORWARD;

                    s->mv_type = MV_TYPE_8X8;

                    s->mb_intra= 0;

                    for(i=0; i<4; i++){

                        s->mv[0][i][0] = s->current_picture.motion_val[0][s->block_index[i]][0];

                        s->mv[0][i][1] = s->current_picture.motion_val[0][s->block_index[i]][1];

                    }

                    break;

                case CANDIDATE_MB_TYPE_DIRECT:

                    if (CONFIG_MPEG4_ENCODER) {

                        s->mv_dir = MV_DIR_FORWARD|MV_DIR_BACKWARD|MV_DIRECT;

                        s->mb_intra= 0;

                        motion_x=s->b_direct_mv_table[xy][0];

                        motion_y=s->b_direct_mv_table[xy][1];

                        ff_mpeg4_set_direct_mv(s, motion_x, motion_y);

                    }

                    break;

                case CANDIDATE_MB_TYPE_DIRECT0:

                    if (CONFIG_MPEG4_ENCODER) {

                        s->mv_dir = MV_DIR_FORWARD|MV_DIR_BACKWARD|MV_DIRECT;

                        s->mb_intra= 0;

                        ff_mpeg4_set_direct_mv(s, 0, 0);

                    }

                    break;

                case CANDIDATE_MB_TYPE_BIDIR:

                    s->mv_dir = MV_DIR_FORWARD | MV_DIR_BACKWARD;

                    s->mb_intra= 0;

                    s->mv[0][0][0] = s->b_bidir_forw_mv_table[xy][0];

                    s->mv[0][0][1] = s->b_bidir_forw_mv_table[xy][1];

                    s->mv[1][0][0] = s->b_bidir_back_mv_table[xy][0];

                    s->mv[1][0][1] = s->b_bidir_back_mv_table[xy][1];

                    break;

                case CANDIDATE_MB_TYPE_BACKWARD:

                    s->mv_dir = MV_DIR_BACKWARD;

                    s->mb_intra= 0;

                    motion_x= s->mv[1][0][0] = s->b_back_mv_table[xy][0];

                    motion_y= s->mv[1][0][1] = s->b_back_mv_table[xy][1];

                    break;

                case CANDIDATE_MB_TYPE_FORWARD:

                    s->mv_dir = MV_DIR_FORWARD;

                    s->mb_intra= 0;

                    motion_x= s->mv[0][0][0] = s->b_forw_mv_table[xy][0];

                    motion_y= s->mv[0][0][1] = s->b_forw_mv_table[xy][1];

                    break;

                case CANDIDATE_MB_TYPE_FORWARD_I:

                    s->mv_dir = MV_DIR_FORWARD;

                    s->mv_type = MV_TYPE_FIELD;

                    s->mb_intra= 0;

                    for(i=0; i<2; i++){

                        j= s->field_select[0][i] = s->b_field_select_table[0][i][xy];

                        s->mv[0][i][0] = s->b_field_mv_table[0][i][j][xy][0];

                        s->mv[0][i][1] = s->b_field_mv_table[0][i][j][xy][1];

                    }

                    break;

                case CANDIDATE_MB_TYPE_BACKWARD_I:

                    s->mv_dir = MV_DIR_BACKWARD;

                    s->mv_type = MV_TYPE_FIELD;

                    s->mb_intra= 0;

                    for(i=0; i<2; i++){

                        j= s->field_select[1][i] = s->b_field_select_table[1][i][xy];

                        s->mv[1][i][0] = s->b_field_mv_table[1][i][j][xy][0];

                        s->mv[1][i][1] = s->b_field_mv_table[1][i][j][xy][1];

                    }

                    break;

                case CANDIDATE_MB_TYPE_BIDIR_I:

                    s->mv_dir = MV_DIR_FORWARD | MV_DIR_BACKWARD;

                    s->mv_type = MV_TYPE_FIELD;

                    s->mb_intra= 0;

                    for(dir=0; dir<2; dir++){

                        for(i=0; i<2; i++){

                            j= s->field_select[dir][i] = s->b_field_select_table[dir][i][xy];

                            s->mv[dir][i][0] = s->b_field_mv_table[dir][i][j][xy][0];

                            s->mv[dir][i][1] = s->b_field_mv_table[dir][i][j][xy][1];

                        }

                    }

                    break;

                default:

                    av_log(s->avctx, AV_LOG_ERROR, "illegal MB type\n");

                }



                encode_mb(s, motion_x, motion_y);



                // RAL: Update last macroblock type

                s->last_mv_dir = s->mv_dir;



                if (CONFIG_H263_ENCODER &&

                    s->out_format == FMT_H263 && s->pict_type!=AV_PICTURE_TYPE_B)

                    ff_h263_update_motion_val(s);



                ff_MPV_decode_mb(s, s->block);

            }



            /* clean the MV table in IPS frames for direct mode in B frames */

            if(s->mb_intra /* && I,P,S_TYPE */){

                s->p_mv_table[xy][0]=0;

                s->p_mv_table[xy][1]=0;

            }



            if(s->flags&CODEC_FLAG_PSNR){

                int w= 16;

                int h= 16;



                if(s->mb_x*16 + 16 > s->width ) w= s->width - s->mb_x*16;

                if(s->mb_y*16 + 16 > s->height) h= s->height- s->mb_y*16;



                s->current_picture.f.error[0] += sse(

                    s, s->new_picture.f.data[0] + s->mb_x*16 + s->mb_y*s->linesize*16,

                    s->dest[0], w, h, s->linesize);

                s->current_picture.f.error[1] += sse(

                    s, s->new_picture.f.data[1] + s->mb_x*8  + s->mb_y*s->uvlinesize*chr_h,

                    s->dest[1], w>>1, h>>s->chroma_y_shift, s->uvlinesize);

                s->current_picture.f.error[2] += sse(

                    s, s->new_picture.f.data[2] + s->mb_x*8  + s->mb_y*s->uvlinesize*chr_h,

                    s->dest[2], w>>1, h>>s->chroma_y_shift, s->uvlinesize);

            }

            if(s->loop_filter){

                if(CONFIG_H263_ENCODER && s->out_format == FMT_H263)

                    ff_h263_loop_filter(s);

            }

            av_dlog(s->avctx, "MB %d %d bits\n",

                    s->mb_x + s->mb_y * s->mb_stride, put_bits_count(&s->pb));

        }

    }



    //not beautiful here but we must write it before flushing so it has to be here

    if (CONFIG_MSMPEG4_ENCODER && s->msmpeg4_version && s->msmpeg4_version<4 && s->pict_type == AV_PICTURE_TYPE_I)

        ff_msmpeg4_encode_ext_header(s);



    write_slice_end(s);



    /* Send the last GOB if RTP */

    if (s->avctx->rtp_callback) {

        int number_mb = (mb_y - s->resync_mb_y)*s->mb_width - s->resync_mb_x;

        pdif = put_bits_ptr(&s->pb) - s->ptr_lastgob;

        /* Call the RTP callback to send the last GOB */

        emms_c();

        s->avctx->rtp_callback(s->avctx, s->ptr_lastgob, pdif, number_mb);

    }



    return 0;

}
