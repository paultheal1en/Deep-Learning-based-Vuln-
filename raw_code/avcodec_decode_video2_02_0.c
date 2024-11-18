int attribute_align_arg avcodec_decode_video2(AVCodecContext *avctx, AVFrame *picture,

                                              int *got_picture_ptr,

                                              const AVPacket *avpkt)

{

    AVCodecInternal *avci = avctx->internal;

    int ret;

    // copy to ensure we do not change avpkt

    AVPacket tmp = *avpkt;



    if (!avctx->codec)

        return AVERROR(EINVAL);

    if (avctx->codec->type != AVMEDIA_TYPE_VIDEO) {

        av_log(avctx, AV_LOG_ERROR, "Invalid media type for video\n");

        return AVERROR(EINVAL);

    }



    *got_picture_ptr = 0;

    if ((avctx->coded_width || avctx->coded_height) && av_image_check_size(avctx->coded_width, avctx->coded_height, 0, avctx))

        return AVERROR(EINVAL);



    avcodec_get_frame_defaults(picture);



    if ((avctx->codec->capabilities & CODEC_CAP_DELAY) || avpkt->size || (avctx->active_thread_type & FF_THREAD_FRAME)) {

        int did_split = av_packet_split_side_data(&tmp);

        ret = apply_param_change(avctx, &tmp);

        if (ret < 0) {

            av_log(avctx, AV_LOG_ERROR, "Error applying parameter changes.\n");

            if (avctx->err_recognition & AV_EF_EXPLODE)

                goto fail;

        }



        avctx->internal->pkt = &tmp;

        if (HAVE_THREADS && avctx->active_thread_type & FF_THREAD_FRAME)

            ret = ff_thread_decode_frame(avctx, picture, got_picture_ptr,

                                         &tmp);

        else {

            ret = avctx->codec->decode(avctx, picture, got_picture_ptr,

                                       &tmp);

            picture->pkt_dts = avpkt->dts;



            if(!avctx->has_b_frames){

                av_frame_set_pkt_pos(picture, avpkt->pos);

            }

            //FIXME these should be under if(!avctx->has_b_frames)

            /* get_buffer is supposed to set frame parameters */

            if (!(avctx->codec->capabilities & CODEC_CAP_DR1)) {

                if (!picture->sample_aspect_ratio.num)    picture->sample_aspect_ratio = avctx->sample_aspect_ratio;

                if (!picture->width)                      picture->width               = avctx->width;

                if (!picture->height)                     picture->height              = avctx->height;

                if (picture->format == AV_PIX_FMT_NONE)   picture->format              = avctx->pix_fmt;

            }

        }

        add_metadata_from_side_data(avctx, picture);



fail:

        emms_c(); //needed to avoid an emms_c() call before every return;



        avctx->internal->pkt = NULL;

        if (did_split) {

            av_packet_free_side_data(&tmp);

            if(ret == tmp.size)

                ret = avpkt->size;

        }



        if (*got_picture_ptr) {

            if (!avctx->refcounted_frames) {

                int err = unrefcount_frame(avci, picture);

                if (err < 0)

                    return err;

            }



            avctx->frame_number++;

            av_frame_set_best_effort_timestamp(picture,

                                               guess_correct_pts(avctx,

                                                                 picture->pkt_pts,

                                                                 picture->pkt_dts));

        } else

            av_frame_unref(picture);

    } else

        ret = 0;



    /* many decoders assign whole AVFrames, thus overwriting extended_data;

     * make sure it's set correctly */

    picture->extended_data = picture->data;



    return ret;

}
