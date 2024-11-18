static int get_cv_color_primaries(AVCodecContext *avctx,

                                  CFStringRef *primaries)

{

    enum AVColorPrimaries pri = avctx->color_primaries;

    switch (pri) {

        case AVCOL_PRI_UNSPECIFIED:

            *primaries = NULL;

            break;



        case AVCOL_PRI_BT709:

            *primaries = kCVImageBufferColorPrimaries_ITU_R_709_2;

            break;



        case AVCOL_PRI_BT2020:

            *primaries = kCVImageBufferColorPrimaries_ITU_R_2020;

            break;



        default:

            av_log(avctx, AV_LOG_ERROR, "Color primaries %s is not supported.\n", av_color_primaries_name(pri));

            *primaries = NULL;

            return -1;

    }



    return 0;

}
