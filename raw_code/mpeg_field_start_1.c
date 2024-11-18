static int mpeg_field_start(MpegEncContext *s, const uint8_t *buf, int buf_size)

{

    AVCodecContext *avctx = s->avctx;

    Mpeg1Context *s1      = (Mpeg1Context *) s;



    /* start frame decoding */

    if (s->first_field || s->picture_structure == PICT_FRAME) {

        AVFrameSideData *pan_scan;



        if (ff_MPV_frame_start(s, avctx) < 0)

            return -1;



        ff_mpeg_er_frame_start(s);



        /* first check if we must repeat the frame */

        s->current_picture_ptr->f.repeat_pict = 0;

        if (s->repeat_first_field) {

            if (s->progressive_sequence) {

                if (s->top_field_first)

                    s->current_picture_ptr->f.repeat_pict = 4;

                else

                    s->current_picture_ptr->f.repeat_pict = 2;

            } else if (s->progressive_frame) {

                s->current_picture_ptr->f.repeat_pict = 1;

            }

        }



        pan_scan = av_frame_new_side_data(&s->current_picture_ptr->f,

                                          AV_FRAME_DATA_PANSCAN,

                                          sizeof(s1->pan_scan));

        if (!pan_scan)

            return AVERROR(ENOMEM);

        memcpy(pan_scan->data, &s1->pan_scan, sizeof(s1->pan_scan));



        if (s1->a53_caption) {

            AVFrameSideData *sd = av_frame_new_side_data(

                &s->current_picture_ptr->f, AV_FRAME_DATA_A53_CC,

                s1->a53_caption_size);

            if (sd)

                memcpy(sd->data, s1->a53_caption, s1->a53_caption_size);

            av_freep(&s1->a53_caption);

        }



        if (s1->has_stereo3d) {

            AVStereo3D *stereo = av_stereo3d_create_side_data(&s->current_picture_ptr->f);

            if (!stereo)

                return AVERROR(ENOMEM);



            *stereo = s1->stereo3d;

            s1->has_stereo3d = 0;

        }

        if (HAVE_THREADS && (avctx->active_thread_type & FF_THREAD_FRAME))

            ff_thread_finish_setup(avctx);

    } else { // second field

        int i;



        if (!s->current_picture_ptr) {

            av_log(s->avctx, AV_LOG_ERROR, "first field missing\n");

            return -1;

        }



        if (s->avctx->hwaccel &&

            (s->avctx->slice_flags & SLICE_FLAG_ALLOW_FIELD)) {

            if (s->avctx->hwaccel->end_frame(s->avctx) < 0)

                av_log(avctx, AV_LOG_ERROR,

                       "hardware accelerator failed to decode first field\n");

        }



        for (i = 0; i < 4; i++) {

            s->current_picture.f.data[i] = s->current_picture_ptr->f.data[i];

            if (s->picture_structure == PICT_BOTTOM_FIELD)

                s->current_picture.f.data[i] +=

                    s->current_picture_ptr->f.linesize[i];

        }

    }



    if (avctx->hwaccel) {

        if (avctx->hwaccel->start_frame(avctx, buf, buf_size) < 0)

            return -1;

    }



#if FF_API_XVMC

FF_DISABLE_DEPRECATION_WARNINGS

// MPV_frame_start will call this function too,

// but we need to call it on every field

    if (CONFIG_MPEG_XVMC_DECODER && s->avctx->xvmc_acceleration)

        if (ff_xvmc_field_start(s, avctx) < 0)

            return -1;

FF_ENABLE_DEPRECATION_WARNINGS

#endif /* FF_API_XVMC */



    return 0;

}
