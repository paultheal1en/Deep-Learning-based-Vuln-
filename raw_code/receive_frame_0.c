static inline CopyRet receive_frame(AVCodecContext *avctx,

                                    void *data, int *got_frame)

{

    BC_STATUS ret;

    BC_DTS_PROC_OUT output = {

        .PicInfo.width  = avctx->width,

        .PicInfo.height = avctx->height,

    };

    CHDContext *priv = avctx->priv_data;

    HANDLE dev       = priv->dev;



    *got_frame = 0;



    // Request decoded data from the driver

    ret = DtsProcOutputNoCopy(dev, OUTPUT_PROC_TIMEOUT, &output);

    if (ret == BC_STS_FMT_CHANGE) {

        av_log(avctx, AV_LOG_VERBOSE, "CrystalHD: Initial format change\n");

        avctx->width  = output.PicInfo.width;

        avctx->height = output.PicInfo.height;

        switch ( output.PicInfo.aspect_ratio ) {

        case vdecAspectRatioSquare:

            avctx->sample_aspect_ratio = (AVRational) {  1,  1};

            break;

        case vdecAspectRatio12_11:

            avctx->sample_aspect_ratio = (AVRational) { 12, 11};

            break;

        case vdecAspectRatio10_11:

            avctx->sample_aspect_ratio = (AVRational) { 10, 11};

            break;

        case vdecAspectRatio16_11:

            avctx->sample_aspect_ratio = (AVRational) { 16, 11};

            break;

        case vdecAspectRatio40_33:

            avctx->sample_aspect_ratio = (AVRational) { 40, 33};

            break;

        case vdecAspectRatio24_11:

            avctx->sample_aspect_ratio = (AVRational) { 24, 11};

            break;

        case vdecAspectRatio20_11:

            avctx->sample_aspect_ratio = (AVRational) { 20, 11};

            break;

        case vdecAspectRatio32_11:

            avctx->sample_aspect_ratio = (AVRational) { 32, 11};

            break;

        case vdecAspectRatio80_33:

            avctx->sample_aspect_ratio = (AVRational) { 80, 33};

            break;

        case vdecAspectRatio18_11:

            avctx->sample_aspect_ratio = (AVRational) { 18, 11};

            break;

        case vdecAspectRatio15_11:

            avctx->sample_aspect_ratio = (AVRational) { 15, 11};

            break;

        case vdecAspectRatio64_33:

            avctx->sample_aspect_ratio = (AVRational) { 64, 33};

            break;

        case vdecAspectRatio160_99:

            avctx->sample_aspect_ratio = (AVRational) {160, 99};

            break;

        case vdecAspectRatio4_3:

            avctx->sample_aspect_ratio = (AVRational) {  4,  3};

            break;

        case vdecAspectRatio16_9:

            avctx->sample_aspect_ratio = (AVRational) { 16,  9};

            break;

        case vdecAspectRatio221_1:

            avctx->sample_aspect_ratio = (AVRational) {221,  1};

            break;

        }

        return RET_OK;

    } else if (ret == BC_STS_SUCCESS) {

        int copy_ret = -1;

        if (output.PoutFlags & BC_POUT_FLAGS_PIB_VALID) {

            if (avctx->codec->id == AV_CODEC_ID_MPEG4 &&

                output.PicInfo.timeStamp == 0 && priv->bframe_bug) {

                if (!priv->bframe_bug) {

                    av_log(avctx, AV_LOG_VERBOSE,

                           "CrystalHD: Not returning packed frame twice.\n");

                }

                DtsReleaseOutputBuffs(dev, NULL, FALSE);

                return RET_COPY_AGAIN;

            }



            print_frame_info(priv, &output);



            copy_ret = copy_frame(avctx, &output, data, got_frame);

        } else {

            /*

             * An invalid frame has been consumed.

             */

            av_log(avctx, AV_LOG_ERROR, "CrystalHD: ProcOutput succeeded with "

                                        "invalid PIB\n");

            copy_ret = RET_OK;

        }

        DtsReleaseOutputBuffs(dev, NULL, FALSE);



        return copy_ret;

    } else if (ret == BC_STS_BUSY) {

        return RET_OK;

    } else {

        av_log(avctx, AV_LOG_ERROR, "CrystalHD: ProcOutput failed %d\n", ret);

        return RET_ERROR;

    }

}
