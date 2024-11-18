static int device_try_init(AVFormatContext *ctx,

                           enum AVPixelFormat pix_fmt,

                           int *width,

                           int *height,

                           uint32_t *desired_format,

                           enum AVCodecID *codec_id)

{

    int ret, i;



    *desired_format = avpriv_fmt_ff2v4l(pix_fmt, ctx->video_codec_id);



    if (*desired_format) {

        ret = device_init(ctx, width, height, *desired_format);

        if (ret < 0) {

            *desired_format = 0;

            if (ret != AVERROR(EINVAL))

                return ret;

        }

    }



    if (!*desired_format) {

        for (i = 0; avpriv_fmt_conversion_table[i].codec_id != AV_CODEC_ID_NONE; i++) {

            if (ctx->video_codec_id == AV_CODEC_ID_NONE ||

                avpriv_fmt_conversion_table[i].codec_id == ctx->video_codec_id) {

                av_log(ctx, AV_LOG_DEBUG, "Trying to set codec:%s pix_fmt:%s\n",

                       avcodec_get_name(avpriv_fmt_conversion_table[i].codec_id),

                       (char *)av_x_if_null(av_get_pix_fmt_name(avpriv_fmt_conversion_table[i].ff_fmt), "none"));



                *desired_format = avpriv_fmt_conversion_table[i].v4l2_fmt;

                ret = device_init(ctx, width, height, *desired_format);

                if (ret >= 0)

                    break;

                else if (ret != AVERROR(EINVAL))

                    return ret;

                *desired_format = 0;

            }

        }



        if (*desired_format == 0) {

            av_log(ctx, AV_LOG_ERROR, "Cannot find a proper format for "

                   "codec '%s' (id %d), pixel format '%s' (id %d)\n",

                   avcodec_get_name(ctx->video_codec_id), ctx->video_codec_id,

                   (char *)av_x_if_null(av_get_pix_fmt_name(pix_fmt), "none"), pix_fmt);

            ret = AVERROR(EINVAL);

        }

    }



    *codec_id = avpriv_fmt_v4l2codec(*desired_format);

    av_assert0(*codec_id != AV_CODEC_ID_NONE);

    return ret;

}
