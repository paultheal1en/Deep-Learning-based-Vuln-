int ff_rv34_decode_frame(AVCodecContext *avctx,

                            void *data, int *got_picture_ptr,

                            AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    RV34DecContext *r = avctx->priv_data;

    MpegEncContext *s = &r->s;

    AVFrame *pict = data;

    SliceInfo si;

    int i, ret;

    int slice_count;

    const uint8_t *slices_hdr = NULL;

    int last = 0;



    /* no supplementary picture */

    if (buf_size == 0) {

        /* special case for last picture */

        if (s->low_delay==0 && s->next_picture_ptr) {

            if ((ret = av_frame_ref(pict, &s->next_picture_ptr->f)) < 0)

                return ret;

            s->next_picture_ptr = NULL;



            *got_picture_ptr = 1;

        }

        return 0;

    }



    if(!avctx->slice_count){

        slice_count = (*buf++) + 1;

        slices_hdr = buf + 4;

        buf += 8 * slice_count;

        buf_size -= 1 + 8 * slice_count;

    }else

        slice_count = avctx->slice_count;



    //parse first slice header to check whether this frame can be decoded

    if(get_slice_offset(avctx, slices_hdr, 0) < 0 ||

       get_slice_offset(avctx, slices_hdr, 0) > buf_size){

        av_log(avctx, AV_LOG_ERROR, "Slice offset is invalid\n");

        return AVERROR_INVALIDDATA;

    }

    init_get_bits(&s->gb, buf+get_slice_offset(avctx, slices_hdr, 0), (buf_size-get_slice_offset(avctx, slices_hdr, 0))*8);

    if(r->parse_slice_header(r, &r->s.gb, &si) < 0 || si.start){

        av_log(avctx, AV_LOG_ERROR, "First slice header is incorrect\n");

        return AVERROR_INVALIDDATA;

    }

    if ((!s->last_picture_ptr || !s->last_picture_ptr->f.data[0]) &&

        si.type == AV_PICTURE_TYPE_B) {

        av_log(avctx, AV_LOG_ERROR, "Invalid decoder state: B-frame without "

               "reference data.\n");

        return AVERROR_INVALIDDATA;

    }

    if(   (avctx->skip_frame >= AVDISCARD_NONREF && si.type==AV_PICTURE_TYPE_B)

       || (avctx->skip_frame >= AVDISCARD_NONKEY && si.type!=AV_PICTURE_TYPE_I)

       ||  avctx->skip_frame >= AVDISCARD_ALL)

        return avpkt->size;



    /* first slice */

    if (si.start == 0) {

        if (s->mb_num_left > 0) {

            av_log(avctx, AV_LOG_ERROR, "New frame but still %d MB left.",

                   s->mb_num_left);

            ff_er_frame_end(&s->er);

            ff_MPV_frame_end(s);

        }



        if (s->width != si.width || s->height != si.height) {

            int err;



            av_log(s->avctx, AV_LOG_WARNING, "Changing dimensions to %dx%d\n",

                   si.width, si.height);



            s->width  = si.width;

            s->height = si.height;



            err = ff_set_dimensions(s->avctx, s->width, s->height);

            if (err < 0)

                return err;



            if ((err = ff_MPV_common_frame_size_change(s)) < 0)

                return err;

            if ((err = rv34_decoder_realloc(r)) < 0)

                return err;

        }

        s->pict_type = si.type ? si.type : AV_PICTURE_TYPE_I;

        if (ff_MPV_frame_start(s, s->avctx) < 0)

            return -1;

        ff_mpeg_er_frame_start(s);

        if (!r->tmp_b_block_base) {

            int i;



            r->tmp_b_block_base = av_malloc(s->linesize * 48);

            for (i = 0; i < 2; i++)

                r->tmp_b_block_y[i] = r->tmp_b_block_base

                                      + i * 16 * s->linesize;

            for (i = 0; i < 4; i++)

                r->tmp_b_block_uv[i] = r->tmp_b_block_base + 32 * s->linesize

                                       + (i >> 1) * 8 * s->uvlinesize

                                       + (i &  1) * 16;

        }

        r->cur_pts = si.pts;

        if (s->pict_type != AV_PICTURE_TYPE_B) {

            r->last_pts = r->next_pts;

            r->next_pts = r->cur_pts;

        } else {

            int refdist = GET_PTS_DIFF(r->next_pts, r->last_pts);

            int dist0   = GET_PTS_DIFF(r->cur_pts,  r->last_pts);

            int dist1   = GET_PTS_DIFF(r->next_pts, r->cur_pts);



            if(!refdist){

                r->mv_weight1 = r->mv_weight2 = r->weight1 = r->weight2 = 8192;

                r->scaled_weight = 0;

            }else{

                r->mv_weight1 = (dist0 << 14) / refdist;

                r->mv_weight2 = (dist1 << 14) / refdist;

                if((r->mv_weight1|r->mv_weight2) & 511){

                    r->weight1 = r->mv_weight1;

                    r->weight2 = r->mv_weight2;

                    r->scaled_weight = 0;

                }else{

                    r->weight1 = r->mv_weight1 >> 9;

                    r->weight2 = r->mv_weight2 >> 9;

                    r->scaled_weight = 1;

                }

            }

        }

        s->mb_x = s->mb_y = 0;

        ff_thread_finish_setup(s->avctx);

    } else if (HAVE_THREADS &&

               (s->avctx->active_thread_type & FF_THREAD_FRAME)) {

        av_log(s->avctx, AV_LOG_ERROR, "Decoder needs full frames in frame "

               "multithreading mode (start MB is %d).\n", si.start);

        return AVERROR_INVALIDDATA;

    }



    for(i = 0; i < slice_count; i++){

        int offset = get_slice_offset(avctx, slices_hdr, i);

        int size;

        if(i+1 == slice_count)

            size = buf_size - offset;

        else

            size = get_slice_offset(avctx, slices_hdr, i+1) - offset;



        if(offset < 0 || offset > buf_size){

            av_log(avctx, AV_LOG_ERROR, "Slice offset is invalid\n");

            break;

        }



        r->si.end = s->mb_width * s->mb_height;

        s->mb_num_left = r->s.mb_x + r->s.mb_y*r->s.mb_width - r->si.start;



        if(i+1 < slice_count){

            if (get_slice_offset(avctx, slices_hdr, i+1) < 0 ||

                get_slice_offset(avctx, slices_hdr, i+1) > buf_size) {

                av_log(avctx, AV_LOG_ERROR, "Slice offset is invalid\n");

                break;

            }

            init_get_bits(&s->gb, buf+get_slice_offset(avctx, slices_hdr, i+1), (buf_size-get_slice_offset(avctx, slices_hdr, i+1))*8);

            if(r->parse_slice_header(r, &r->s.gb, &si) < 0){

                if(i+2 < slice_count)

                    size = get_slice_offset(avctx, slices_hdr, i+2) - offset;

                else

                    size = buf_size - offset;

            }else

                r->si.end = si.start;

        }

        if (size < 0 || size > buf_size - offset) {

            av_log(avctx, AV_LOG_ERROR, "Slice size is invalid\n");

            break;

        }

        last = rv34_decode_slice(r, r->si.end, buf + offset, size);

        if(last)

            break;

    }



    if (s->current_picture_ptr) {

        if (last) {

            if(r->loop_filter)

                r->loop_filter(r, s->mb_height - 1);



            ret = finish_frame(avctx, pict);

            if (ret < 0)

                return ret;

            *got_picture_ptr = ret;

        } else if (HAVE_THREADS &&

                   (s->avctx->active_thread_type & FF_THREAD_FRAME)) {

            av_log(avctx, AV_LOG_INFO, "marking unfished frame as finished\n");

            /* always mark the current frame as finished, frame-mt supports

             * only complete frames */

            ff_er_frame_end(&s->er);

            ff_MPV_frame_end(s);

            s->mb_num_left = 0;

            ff_thread_report_progress(&s->current_picture_ptr->tf, INT_MAX, 0);

            return AVERROR_INVALIDDATA;

        }

    }



    return avpkt->size;

}
