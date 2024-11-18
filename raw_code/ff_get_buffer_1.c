int ff_get_buffer(AVCodecContext *avctx, AVFrame *frame, int flags)

{

    int ret;



    switch (avctx->codec_type) {

    case AVMEDIA_TYPE_VIDEO:

        if (!frame->width)

            frame->width               = avctx->width;

        if (!frame->height)

            frame->height              = avctx->height;

        if (frame->format < 0)

            frame->format              = avctx->pix_fmt;

        if (!frame->sample_aspect_ratio.num)

            frame->sample_aspect_ratio = avctx->sample_aspect_ratio;



        if ((ret = av_image_check_size(avctx->width, avctx->height, 0, avctx)) < 0)

            return ret;

        break;

    case AVMEDIA_TYPE_AUDIO:

        if (!frame->sample_rate)

            frame->sample_rate    = avctx->sample_rate;

        if (frame->format < 0)

            frame->format         = avctx->sample_fmt;

        if (!frame->channel_layout) {

            if (avctx->channel_layout) {

                 if (av_get_channel_layout_nb_channels(avctx->channel_layout) !=

                     avctx->channels) {

                     av_log(avctx, AV_LOG_ERROR, "Inconsistent channel "

                            "configuration.\n");

                     return AVERROR(EINVAL);

                 }



                frame->channel_layout = avctx->channel_layout;

            } else {

                if (avctx->channels > FF_SANE_NB_CHANNELS) {

                    av_log(avctx, AV_LOG_ERROR, "Too many channels: %d.\n",

                           avctx->channels);

                    return AVERROR(ENOSYS);

                }



                frame->channel_layout = av_get_default_channel_layout(avctx->channels);

                if (!frame->channel_layout)

                    frame->channel_layout = (1ULL << avctx->channels) - 1;

            }

        }

        break;

    default: return AVERROR(EINVAL);

    }



    frame->pkt_pts = avctx->pkt ? avctx->pkt->pts : AV_NOPTS_VALUE;

    frame->reordered_opaque = avctx->reordered_opaque;



#if FF_API_GET_BUFFER

    /*

     * Wrap an old get_buffer()-allocated buffer in an bunch of AVBuffers.

     * We wrap each plane in its own AVBuffer. Each of those has a reference to

     * a dummy AVBuffer as its private data, unreffing it on free.

     * When all the planes are freed, the dummy buffer's free callback calls

     * release_buffer().

     */

    if (avctx->get_buffer) {

        CompatReleaseBufPriv *priv = NULL;

        AVBufferRef *dummy_buf = NULL;

        int planes, i, ret;



        if (flags & AV_GET_BUFFER_FLAG_REF)

            frame->reference    = 1;



        ret = avctx->get_buffer(avctx, frame);

        if (ret < 0)

            return ret;



        /* return if the buffers are already set up

         * this would happen e.g. when a custom get_buffer() calls

         * avcodec_default_get_buffer

         */

        if (frame->buf[0])

            return 0;



        priv = av_mallocz(sizeof(*priv));

        if (!priv) {

            ret = AVERROR(ENOMEM);

            goto fail;

        }

        priv->avctx = *avctx;

        priv->frame = *frame;



        dummy_buf = av_buffer_create(NULL, 0, compat_free_buffer, priv, 0);

        if (!dummy_buf) {

            ret = AVERROR(ENOMEM);

            goto fail;

        }



#define WRAP_PLANE(ref_out, data, data_size)                            \

do {                                                                    \

    AVBufferRef *dummy_ref = av_buffer_ref(dummy_buf);                  \

    if (!dummy_ref) {                                                   \

        ret = AVERROR(ENOMEM);                                          \

        goto fail;                                                      \

    }                                                                   \

    ref_out = av_buffer_create(data, data_size, compat_release_buffer,  \

                               dummy_ref, 0);                           \

    if (!ref_out) {                                                     \

        av_frame_unref(frame);                                          \

        ret = AVERROR(ENOMEM);                                          \

        goto fail;                                                      \

    }                                                                   \

} while (0)



        if (avctx->codec_type == AVMEDIA_TYPE_VIDEO) {

            const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(frame->format);



            if (!desc) {

                ret = AVERROR(EINVAL);

                goto fail;

            }

            planes = (desc->flags & PIX_FMT_PLANAR) ? desc->nb_components : 1;



            for (i = 0; i < planes; i++) {

                int h_shift    = (i == 1 || i == 2) ? desc->log2_chroma_h : 0;

                int plane_size = (frame->width >> h_shift) * frame->linesize[i];



                WRAP_PLANE(frame->buf[i], frame->data[i], plane_size);

            }

        } else {

            int planar = av_sample_fmt_is_planar(frame->format);

            planes = planar ? avctx->channels : 1;



            if (planes > FF_ARRAY_ELEMS(frame->buf)) {

                frame->nb_extended_buf = planes - FF_ARRAY_ELEMS(frame->buf);

                frame->extended_buf = av_malloc(sizeof(*frame->extended_buf) *

                                                frame->nb_extended_buf);

                if (!frame->extended_buf) {

                    ret = AVERROR(ENOMEM);

                    goto fail;

                }

            }



            for (i = 0; i < FFMIN(planes, FF_ARRAY_ELEMS(frame->buf)); i++)

                WRAP_PLANE(frame->buf[i], frame->extended_data[i], frame->linesize[0]);



            for (i = 0; i < planes - FF_ARRAY_ELEMS(frame->buf); i++)

                WRAP_PLANE(frame->extended_buf[i],

                           frame->extended_data[i + FF_ARRAY_ELEMS(frame->buf)],

                           frame->linesize[0]);

        }



        av_buffer_unref(&dummy_buf);



        return 0;



fail:

        avctx->release_buffer(avctx, frame);

        av_freep(&priv);

        av_buffer_unref(&dummy_buf);

        return ret;

    }

#endif



    return avctx->get_buffer2(avctx, frame, flags);

}
