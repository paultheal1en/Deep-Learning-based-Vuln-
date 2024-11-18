static int try_decode_video_frame(AVCodecContext *codec_ctx, AVPacket *pkt, int decode)

{

    int ret = 0;

    int got_frame = 0;

    AVFrame *frame = NULL;

    int skip_frame = codec_ctx->skip_frame;



    if (!avcodec_is_open(codec_ctx)) {

        const AVCodec *codec = avcodec_find_decoder(codec_ctx->codec_id);



        ret = avcodec_open2(codec_ctx, codec, NULL);

        if (ret < 0) {

            av_log(codec_ctx, AV_LOG_ERROR, "Failed to open codec\n");

            goto end;

        }

    }



    frame = av_frame_alloc();

    if (!frame) {

        av_log(NULL, AV_LOG_ERROR, "Failed to allocate frame\n");

        goto end;

    }



    if (!decode && codec_ctx->codec->caps_internal & FF_CODEC_CAP_SKIP_FRAME_FILL_PARAM) {

        codec_ctx->skip_frame = AVDISCARD_ALL;

    }



    do {

        ret = avcodec_decode_video2(codec_ctx, frame, &got_frame, pkt);

        av_assert0(decode || (!decode && !got_frame));

        if (ret < 0)

            break;

        pkt->data += ret;

        pkt->size -= ret;



        if (got_frame) {

            break;

        }

    } while (pkt->size > 0);



end:

    codec_ctx->skip_frame = skip_frame;



    av_frame_free(&frame);

    return ret;

}
