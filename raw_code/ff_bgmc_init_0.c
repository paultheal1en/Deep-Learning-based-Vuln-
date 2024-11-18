int ff_bgmc_init(AVCodecContext *avctx, uint8_t **cf_lut, int **cf_lut_status)

{

    *cf_lut        = av_malloc(sizeof(**cf_lut)        * LUT_BUFF * 16 * LUT_SIZE);

    *cf_lut_status = av_malloc(sizeof(**cf_lut_status) * LUT_BUFF);



    if (!cf_lut || !cf_lut_status) {

        ff_bgmc_end(cf_lut, cf_lut_status);

        av_log(avctx, AV_LOG_ERROR, "Allocating buffer memory failed.\n");

        return AVERROR(ENOMEM);

    } else {

        // initialize lut_status buffer to a value never used to compare against

        memset(*cf_lut_status, -1, sizeof(**cf_lut_status) * LUT_BUFF);

    }



    return 0;

}
