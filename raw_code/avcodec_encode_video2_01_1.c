int attribute_align_arg avcodec_encode_video2(AVCodecContext *avctx,

                                              AVPacket *avpkt,

                                              const AVFrame *frame,

                                              int *got_packet_ptr)

{

    int ret;

    int user_packet = !!avpkt->data;



    *got_packet_ptr = 0;



    if (!(avctx->codec->capabilities & CODEC_CAP_DELAY) && !frame) {

        av_free_packet(avpkt);

        av_init_packet(avpkt);

        avpkt->size     = 0;

        return 0;

    }



    if (av_image_check_size(avctx->width, avctx->height, 0, avctx))

        return AVERROR(EINVAL);



    av_assert0(avctx->codec->encode2);



    ret = avctx->codec->encode2(avctx, avpkt, frame, got_packet_ptr);

    if (!ret) {

        if (!*got_packet_ptr)

            avpkt->size = 0;

        else if (!(avctx->codec->capabilities & CODEC_CAP_DELAY))

            avpkt->pts = avpkt->dts = frame->pts;



        if (!user_packet && avpkt->data) {

            uint8_t *new_data = av_realloc(avpkt->data, avpkt->size);

            if (new_data)

                avpkt->data = new_data;

        }



        avctx->frame_number++;

    }



    if (ret < 0 || !*got_packet_ptr)

        av_free_packet(avpkt);



    emms_c();

    return ret;

}
