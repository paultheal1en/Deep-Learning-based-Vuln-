int attribute_align_arg avcodec_encode_audio2(AVCodecContext *avctx,

                                              AVPacket *avpkt,

                                              const AVFrame *frame,

                                              int *got_packet_ptr)

{

    int ret;

    int user_packet = !!avpkt->data;

    int nb_samples;



    *got_packet_ptr = 0;



    if (!(avctx->codec->capabilities & CODEC_CAP_DELAY) && !frame) {

        av_free_packet(avpkt);

        av_init_packet(avpkt);

        avpkt->size = 0;

        return 0;

    }



    /* check for valid frame size */

    if (frame) {

        nb_samples = frame->nb_samples;

        if (avctx->codec->capabilities & CODEC_CAP_SMALL_LAST_FRAME) {

            if (nb_samples > avctx->frame_size)

                return AVERROR(EINVAL);

        } else if (!(avctx->codec->capabilities & CODEC_CAP_VARIABLE_FRAME_SIZE)) {

            if (nb_samples != avctx->frame_size)

                return AVERROR(EINVAL);

        }

    } else {

        nb_samples = avctx->frame_size;

    }



    if (avctx->codec->encode2) {

        ret = avctx->codec->encode2(avctx, avpkt, frame, got_packet_ptr);

        if (!ret && *got_packet_ptr) {

            if (!(avctx->codec->capabilities & CODEC_CAP_DELAY)) {

                if (avpkt->pts == AV_NOPTS_VALUE)

                    avpkt->pts = frame->pts;

                if (!avpkt->duration)

                    avpkt->duration = ff_samples_to_time_base(avctx,

                                                              frame->nb_samples);

            }

            avpkt->dts = avpkt->pts;

        } else {

            avpkt->size = 0;

        }

    } else {

        /* for compatibility with encoders not supporting encode2(), we need to

           allocate a packet buffer if the user has not provided one or check

           the size otherwise */

        int fs_tmp   = 0;

        int buf_size = avpkt->size;

        if (!user_packet) {

            if (avctx->codec->capabilities & CODEC_CAP_VARIABLE_FRAME_SIZE) {

                av_assert0(av_get_bits_per_sample(avctx->codec_id) != 0);

                buf_size = nb_samples * avctx->channels *

                           av_get_bits_per_sample(avctx->codec_id) / 8;

            } else {

                /* this is a guess as to the required size.

                   if an encoder needs more than this, it should probably

                   implement encode2() */

                buf_size = 2 * avctx->frame_size * avctx->channels *

                           av_get_bytes_per_sample(avctx->sample_fmt);

                buf_size += FF_MIN_BUFFER_SIZE;

            }

        }

        if ((ret = ff_alloc_packet(avpkt, buf_size)))

            return ret;



        /* Encoders using AVCodec.encode() that support

           CODEC_CAP_SMALL_LAST_FRAME require avctx->frame_size to be set to

           the smaller size when encoding the last frame.

           This code can be removed once all encoders supporting

           CODEC_CAP_SMALL_LAST_FRAME use encode2() */

        if ((avctx->codec->capabilities & CODEC_CAP_SMALL_LAST_FRAME) &&

            nb_samples < avctx->frame_size) {

            fs_tmp = avctx->frame_size;

            avctx->frame_size = nb_samples;

        }



        /* encode the frame */

        ret = avctx->codec->encode(avctx, avpkt->data, avpkt->size,

                                   frame ? frame->data[0] : NULL);

        if (ret >= 0) {

            if (!ret) {

                /* no output. if the packet data was allocated by libavcodec,

                   free it */

                if (!user_packet)

                    av_freep(&avpkt->data);

            } else {

                if (avctx->coded_frame)

                    avpkt->pts = avpkt->dts = avctx->coded_frame->pts;

                /* Set duration for final small packet. This can be removed

                   once all encoders supporting CODEC_CAP_SMALL_LAST_FRAME use

                   encode2() */

                if (fs_tmp) {

                    avpkt->duration = ff_samples_to_time_base(avctx,

                                                              avctx->frame_size);

                }

            }

            avpkt->size = ret;

            *got_packet_ptr = (ret > 0);

            ret = 0;

        }



        if (fs_tmp)

            avctx->frame_size = fs_tmp;

    }

    if (!ret) {

        if (!user_packet && avpkt->data) {

            uint8_t *new_data = av_realloc(avpkt->data, avpkt->size);

            if (new_data)

                avpkt->data = new_data;

        }



        avctx->frame_number++;

    }



    if (ret < 0 || !*got_packet_ptr)

        av_free_packet(avpkt);



    /* NOTE: if we add any audio encoders which output non-keyframe packets,

             this needs to be moved to the encoders, but for now we can do it

             here to simplify things */

    avpkt->flags |= AV_PKT_FLAG_KEY;



    return ret;

}
