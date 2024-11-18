int ff_mpv_frame_start(MpegEncContext *s, AVCodecContext *avctx)

{

    int i, ret;

    Picture *pic;

    s->mb_skipped = 0;



    /* mark & release old frames */

    if (s->pict_type != AV_PICTURE_TYPE_B && s->last_picture_ptr &&

        s->last_picture_ptr != s->next_picture_ptr &&

        s->last_picture_ptr->f->buf[0]) {

        ff_mpeg_unref_picture(s->avctx, s->last_picture_ptr);

    }



    /* release forgotten pictures */

    /* if (MPEG-124 / H.263) */

    for (i = 0; i < MAX_PICTURE_COUNT; i++) {

        if (&s->picture[i] != s->last_picture_ptr &&

            &s->picture[i] != s->next_picture_ptr &&

            s->picture[i].reference && !s->picture[i].needs_realloc) {

            ff_mpeg_unref_picture(s->avctx, &s->picture[i]);

        }

    }



    ff_mpeg_unref_picture(s->avctx, &s->current_picture);



    /* release non reference frames */

    for (i = 0; i < MAX_PICTURE_COUNT; i++) {

        if (!s->picture[i].reference)

            ff_mpeg_unref_picture(s->avctx, &s->picture[i]);

    }



    if (s->current_picture_ptr && !s->current_picture_ptr->f->buf[0]) {

        // we already have a unused image

        // (maybe it was set before reading the header)

        pic = s->current_picture_ptr;

    } else {

        i   = ff_find_unused_picture(s->avctx, s->picture, 0);

        if (i < 0) {

            av_log(s->avctx, AV_LOG_ERROR, "no frame buffer available\n");

            return i;

        }

        pic = &s->picture[i];

    }



    pic->reference = 0;

    if (!s->droppable) {

        if (s->pict_type != AV_PICTURE_TYPE_B)

            pic->reference = 3;

    }



    pic->f->coded_picture_number = s->coded_picture_number++;



    if (alloc_picture(s, pic, 0) < 0)

        return -1;



    s->current_picture_ptr = pic;

    // FIXME use only the vars from current_pic

    s->current_picture_ptr->f->top_field_first = s->top_field_first;

    if (s->codec_id == AV_CODEC_ID_MPEG1VIDEO ||

        s->codec_id == AV_CODEC_ID_MPEG2VIDEO) {

        if (s->picture_structure != PICT_FRAME)

            s->current_picture_ptr->f->top_field_first =

                (s->picture_structure == PICT_TOP_FIELD) == s->first_field;

    }

    s->current_picture_ptr->f->interlaced_frame = !s->progressive_frame &&

                                                 !s->progressive_sequence;

    s->current_picture_ptr->field_picture      =  s->picture_structure != PICT_FRAME;



    s->current_picture_ptr->f->pict_type = s->pict_type;

    // if (s->avctx->flags && AV_CODEC_FLAG_QSCALE)

    //     s->current_picture_ptr->quality = s->new_picture_ptr->quality;

    s->current_picture_ptr->f->key_frame = s->pict_type == AV_PICTURE_TYPE_I;



    if ((ret = ff_mpeg_ref_picture(s->avctx, &s->current_picture,

                                   s->current_picture_ptr)) < 0)

        return ret;



    if (s->pict_type != AV_PICTURE_TYPE_B) {

        s->last_picture_ptr = s->next_picture_ptr;

        if (!s->droppable)

            s->next_picture_ptr = s->current_picture_ptr;

    }

    ff_dlog(s->avctx, "L%p N%p C%p L%p N%p C%p type:%d drop:%d\n",

            s->last_picture_ptr, s->next_picture_ptr,s->current_picture_ptr,

            s->last_picture_ptr    ? s->last_picture_ptr->f->data[0]    : NULL,

            s->next_picture_ptr    ? s->next_picture_ptr->f->data[0]    : NULL,

            s->current_picture_ptr ? s->current_picture_ptr->f->data[0] : NULL,

            s->pict_type, s->droppable);



    if ((!s->last_picture_ptr || !s->last_picture_ptr->f->buf[0]) &&

        (s->pict_type != AV_PICTURE_TYPE_I ||

         s->picture_structure != PICT_FRAME)) {

        int h_chroma_shift, v_chroma_shift;

        av_pix_fmt_get_chroma_sub_sample(s->avctx->pix_fmt,

                                         &h_chroma_shift, &v_chroma_shift);

        if (s->pict_type != AV_PICTURE_TYPE_I)

            av_log(avctx, AV_LOG_ERROR,

                   "warning: first frame is no keyframe\n");

        else if (s->picture_structure != PICT_FRAME)

            av_log(avctx, AV_LOG_INFO,

                   "allocate dummy last picture for field based first keyframe\n");



        /* Allocate a dummy frame */

        i = ff_find_unused_picture(s->avctx, s->picture, 0);

        if (i < 0) {

            av_log(s->avctx, AV_LOG_ERROR, "no frame buffer available\n");

            return i;

        }

        s->last_picture_ptr = &s->picture[i];



        s->last_picture_ptr->reference   = 3;

        s->last_picture_ptr->f->pict_type = AV_PICTURE_TYPE_I;



        if (alloc_picture(s, s->last_picture_ptr, 0) < 0) {

            s->last_picture_ptr = NULL;

            return -1;

        }



        memset(s->last_picture_ptr->f->data[0], 0,

               avctx->height * s->last_picture_ptr->f->linesize[0]);

        memset(s->last_picture_ptr->f->data[1], 0x80,

               (avctx->height >> v_chroma_shift) *

               s->last_picture_ptr->f->linesize[1]);

        memset(s->last_picture_ptr->f->data[2], 0x80,

               (avctx->height >> v_chroma_shift) *

               s->last_picture_ptr->f->linesize[2]);



        ff_thread_report_progress(&s->last_picture_ptr->tf, INT_MAX, 0);

        ff_thread_report_progress(&s->last_picture_ptr->tf, INT_MAX, 1);

    }

    if ((!s->next_picture_ptr || !s->next_picture_ptr->f->buf[0]) &&

        s->pict_type == AV_PICTURE_TYPE_B) {

        /* Allocate a dummy frame */

        i = ff_find_unused_picture(s->avctx, s->picture, 0);

        if (i < 0) {

            av_log(s->avctx, AV_LOG_ERROR, "no frame buffer available\n");

            return i;

        }

        s->next_picture_ptr = &s->picture[i];



        s->next_picture_ptr->reference   = 3;

        s->next_picture_ptr->f->pict_type = AV_PICTURE_TYPE_I;



        if (alloc_picture(s, s->next_picture_ptr, 0) < 0) {

            s->next_picture_ptr = NULL;

            return -1;

        }

        ff_thread_report_progress(&s->next_picture_ptr->tf, INT_MAX, 0);

        ff_thread_report_progress(&s->next_picture_ptr->tf, INT_MAX, 1);

    }



    if (s->last_picture_ptr) {

        ff_mpeg_unref_picture(s->avctx, &s->last_picture);

        if (s->last_picture_ptr->f->buf[0] &&

            (ret = ff_mpeg_ref_picture(s->avctx, &s->last_picture,

                                       s->last_picture_ptr)) < 0)

            return ret;

    }

    if (s->next_picture_ptr) {

        ff_mpeg_unref_picture(s->avctx, &s->next_picture);

        if (s->next_picture_ptr->f->buf[0] &&

            (ret = ff_mpeg_ref_picture(s->avctx, &s->next_picture,

                                       s->next_picture_ptr)) < 0)

            return ret;

    }



    if (s->pict_type != AV_PICTURE_TYPE_I &&

        !(s->last_picture_ptr && s->last_picture_ptr->f->buf[0])) {

        av_log(s, AV_LOG_ERROR,

               "Non-reference picture received and no reference available\n");

        return AVERROR_INVALIDDATA;

    }



    if (s->picture_structure!= PICT_FRAME) {

        int i;

        for (i = 0; i < 4; i++) {

            if (s->picture_structure == PICT_BOTTOM_FIELD) {

                s->current_picture.f->data[i] +=

                    s->current_picture.f->linesize[i];

            }

            s->current_picture.f->linesize[i] *= 2;

            s->last_picture.f->linesize[i]    *= 2;

            s->next_picture.f->linesize[i]    *= 2;

        }

    }



    /* set dequantizer, we can't do it during init as

     * it might change for MPEG-4 and we can't do it in the header

     * decode as init is not called for MPEG-4 there yet */

    if (s->mpeg_quant || s->codec_id == AV_CODEC_ID_MPEG2VIDEO) {

        s->dct_unquantize_intra = s->dct_unquantize_mpeg2_intra;

        s->dct_unquantize_inter = s->dct_unquantize_mpeg2_inter;

    } else if (s->out_format == FMT_H263 || s->out_format == FMT_H261) {

        s->dct_unquantize_intra = s->dct_unquantize_h263_intra;

        s->dct_unquantize_inter = s->dct_unquantize_h263_inter;

    } else {

        s->dct_unquantize_intra = s->dct_unquantize_mpeg1_intra;

        s->dct_unquantize_inter = s->dct_unquantize_mpeg1_inter;

    }



#if FF_API_XVMC

FF_DISABLE_DEPRECATION_WARNINGS

    if (CONFIG_MPEG_XVMC_DECODER && s->avctx->xvmc_acceleration)

        return ff_xvmc_field_start(s, avctx);

FF_ENABLE_DEPRECATION_WARNINGS

#endif /* FF_API_XVMC */



    return 0;

}
