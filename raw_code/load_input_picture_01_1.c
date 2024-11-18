static int load_input_picture(MpegEncContext *s, const AVFrame *pic_arg)

{

    Picture *pic = NULL;

    int64_t pts;

    int i, display_picture_number = 0, ret;

    int encoding_delay = s->max_b_frames ? s->max_b_frames

                                         : (s->low_delay ? 0 : 1);

    int flush_offset = 1;

    int direct = 1;



    if (pic_arg) {

        pts = pic_arg->pts;

        display_picture_number = s->input_picture_number++;



        if (pts != AV_NOPTS_VALUE) {

            if (s->user_specified_pts != AV_NOPTS_VALUE) {

                int64_t last = s->user_specified_pts;



                if (pts <= last) {

                    av_log(s->avctx, AV_LOG_ERROR,

                           "Invalid pts (%"PRId64") <= last (%"PRId64")\n",

                           pts, last);

                    return AVERROR(EINVAL);

                }



                if (!s->low_delay && display_picture_number == 1)

                    s->dts_delta = pts - last;

            }

            s->user_specified_pts = pts;

        } else {

            if (s->user_specified_pts != AV_NOPTS_VALUE) {

                s->user_specified_pts =

                pts = s->user_specified_pts + 1;

                av_log(s->avctx, AV_LOG_INFO,

                       "Warning: AVFrame.pts=? trying to guess (%"PRId64")\n",

                       pts);

            } else {

                pts = display_picture_number;

            }

        }



        if (!pic_arg->buf[0] ||

            pic_arg->linesize[0] != s->linesize ||

            pic_arg->linesize[1] != s->uvlinesize ||

            pic_arg->linesize[2] != s->uvlinesize)

            direct = 0;

        if ((s->width & 15) || (s->height & 15))

            direct = 0;

        if (((intptr_t)(pic_arg->data[0])) & (STRIDE_ALIGN-1))

            direct = 0;

        if (s->linesize & (STRIDE_ALIGN-1))

            direct = 0;



        ff_dlog(s->avctx, "%d %d %"PTRDIFF_SPECIFIER" %"PTRDIFF_SPECIFIER"\n", pic_arg->linesize[0],

                pic_arg->linesize[1], s->linesize, s->uvlinesize);



        i = ff_find_unused_picture(s->avctx, s->picture, direct);

        if (i < 0)

            return i;



        pic = &s->picture[i];

        pic->reference = 3;



        if (direct) {

            if ((ret = av_frame_ref(pic->f, pic_arg)) < 0)

                return ret;

        }

        ret = alloc_picture(s, pic, direct);

        if (ret < 0)

            return ret;



        if (!direct) {

            if (pic->f->data[0] + INPLACE_OFFSET == pic_arg->data[0] &&

                pic->f->data[1] + INPLACE_OFFSET == pic_arg->data[1] &&

                pic->f->data[2] + INPLACE_OFFSET == pic_arg->data[2]) {

                // empty

            } else {

                int h_chroma_shift, v_chroma_shift;

                av_pix_fmt_get_chroma_sub_sample(s->avctx->pix_fmt,

                                                 &h_chroma_shift,

                                                 &v_chroma_shift);



                for (i = 0; i < 3; i++) {

                    int src_stride = pic_arg->linesize[i];

                    int dst_stride = i ? s->uvlinesize : s->linesize;

                    int h_shift = i ? h_chroma_shift : 0;

                    int v_shift = i ? v_chroma_shift : 0;

                    int w = s->width  >> h_shift;

                    int h = s->height >> v_shift;

                    uint8_t *src = pic_arg->data[i];

                    uint8_t *dst = pic->f->data[i];

                    int vpad = 16;



                    if (   s->codec_id == AV_CODEC_ID_MPEG2VIDEO

                        && !s->progressive_sequence

                        && FFALIGN(s->height, 32) - s->height > 16)

                        vpad = 32;



                    if (!s->avctx->rc_buffer_size)

                        dst += INPLACE_OFFSET;



                    if (src_stride == dst_stride)

                        memcpy(dst, src, src_stride * h);

                    else {

                        int h2 = h;

                        uint8_t *dst2 = dst;

                        while (h2--) {

                            memcpy(dst2, src, w);

                            dst2 += dst_stride;

                            src += src_stride;

                        }

                    }

                    if ((s->width & 15) || (s->height & (vpad-1))) {

                        s->mpvencdsp.draw_edges(dst, dst_stride,

                                                w, h,

                                                16 >> h_shift,

                                                vpad >> v_shift,

                                                EDGE_BOTTOM);

                    }

                }


            }

        }

        ret = av_frame_copy_props(pic->f, pic_arg);

        if (ret < 0)

            return ret;



        pic->f->display_picture_number = display_picture_number;

        pic->f->pts = pts; // we set this here to avoid modifying pic_arg

    } else {

        /* Flushing: When we have not received enough input frames,

         * ensure s->input_picture[0] contains the first picture */

        for (flush_offset = 0; flush_offset < encoding_delay + 1; flush_offset++)

            if (s->input_picture[flush_offset])

                break;



        if (flush_offset <= 1)

            flush_offset = 1;

        else

            encoding_delay = encoding_delay - flush_offset + 1;

    }



    /* shift buffer entries */

    for (i = flush_offset; i < MAX_PICTURE_COUNT /*s->encoding_delay + 1*/; i++)

        s->input_picture[i - flush_offset] = s->input_picture[i];



    s->input_picture[encoding_delay] = (Picture*) pic;



    return 0;

}