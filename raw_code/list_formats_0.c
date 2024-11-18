static void list_formats(AVFormatContext *ctx, int type)

{

    const struct video_data *s = ctx->priv_data;

    struct v4l2_fmtdesc vfd = { .type = V4L2_BUF_TYPE_VIDEO_CAPTURE };



    while(!v4l2_ioctl(s->fd, VIDIOC_ENUM_FMT, &vfd)) {

        enum AVCodecID codec_id = avpriv_fmt_v4l2codec(vfd.pixelformat);

        enum AVPixelFormat pix_fmt = avpriv_fmt_v4l2ff(vfd.pixelformat, codec_id);



        vfd.index++;



        if (!(vfd.flags & V4L2_FMT_FLAG_COMPRESSED) &&

            type & V4L_RAWFORMATS) {

            const char *fmt_name = av_get_pix_fmt_name(pix_fmt);

            av_log(ctx, AV_LOG_INFO, "Raw       : %9s : %20s :",

                   fmt_name ? fmt_name : "Unsupported",

                   vfd.description);

        } else if (vfd.flags & V4L2_FMT_FLAG_COMPRESSED &&

                   type & V4L_COMPFORMATS) {

            AVCodec *codec = avcodec_find_decoder(codec_id);

            av_log(ctx, AV_LOG_INFO, "Compressed: %9s : %20s :",

                   codec ? codec->name : "Unsupported",

                   vfd.description);

        } else {

            continue;

        }



#ifdef V4L2_FMT_FLAG_EMULATED

        if (vfd.flags & V4L2_FMT_FLAG_EMULATED)

            av_log(ctx, AV_LOG_INFO, " Emulated :");

#endif

#if HAVE_STRUCT_V4L2_FRMIVALENUM_DISCRETE

        list_framesizes(ctx, vfd.pixelformat);

#endif

        av_log(ctx, AV_LOG_INFO, "\n");

    }

}
