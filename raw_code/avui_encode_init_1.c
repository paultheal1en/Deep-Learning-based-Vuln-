static av_cold int avui_encode_init(AVCodecContext *avctx)

{

    avctx->coded_frame = av_frame_alloc();



    if (avctx->width != 720 || avctx->height != 486 && avctx->height != 576) {

        av_log(avctx, AV_LOG_ERROR, "Only 720x486 and 720x576 are supported.\n");

        return AVERROR(EINVAL);

    }

    if (!avctx->coded_frame) {

        av_log(avctx, AV_LOG_ERROR, "Could not allocate frame.\n");

        return AVERROR(ENOMEM);

    }

    if (!(avctx->extradata = av_mallocz(24 + FF_INPUT_BUFFER_PADDING_SIZE)))

        return AVERROR(ENOMEM);

    avctx->extradata_size = 24;

    memcpy(avctx->extradata, "\0\0\0\x18""APRGAPRG0001", 16);

    if (avctx->field_order > AV_FIELD_PROGRESSIVE) {

        avctx->extradata[19] = 2;

    } else {

        avctx->extradata[19] = 1;

    }





    return 0;

}
