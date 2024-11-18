av_cold static int fbdev_read_header(AVFormatContext *avctx,

                                     AVFormatParameters *ap)

{

    FBDevContext *fbdev = avctx->priv_data;

    AVStream *st = NULL;

    enum PixelFormat pix_fmt;

    int ret, flags = O_RDONLY;



    ret = av_parse_video_rate(&fbdev->framerate_q, fbdev->framerate);

    if (ret < 0) {

        av_log(avctx, AV_LOG_ERROR, "Couldn't parse framerate.\n");

        return ret;

    }

#if FF_API_FORMAT_PARAMETERS

    if (ap->time_base.num)

        fbdev->framerate_q = (AVRational){ap->time_base.den, ap->time_base.num};

#endif



    if (!(st = av_new_stream(avctx, 0)))

        return AVERROR(ENOMEM);

    av_set_pts_info(st, 64, 1, 1000000); /* 64 bits pts in microseconds */



    /* NONBLOCK is ignored by the fbdev driver, only set for consistency */

    if (avctx->flags & AVFMT_FLAG_NONBLOCK)

        flags |= O_NONBLOCK;



    if ((fbdev->fd = open(avctx->filename, flags)) == -1) {

        ret = AVERROR(errno);

        av_log(avctx, AV_LOG_ERROR,

               "Could not open framebuffer device '%s': %s\n",

               avctx->filename, strerror(ret));

        return ret;

    }



    if (ioctl(fbdev->fd, FBIOGET_VSCREENINFO, &fbdev->varinfo) < 0) {

        ret = AVERROR(errno);

        av_log(avctx, AV_LOG_ERROR,

               "FBIOGET_VSCREENINFO: %s\n", strerror(errno));

        goto fail;

    }



    if (ioctl(fbdev->fd, FBIOGET_FSCREENINFO, &fbdev->fixinfo) < 0) {

        ret = AVERROR(errno);

        av_log(avctx, AV_LOG_ERROR,

               "FBIOGET_FSCREENINFO: %s\n", strerror(errno));

        goto fail;

    }



    pix_fmt = get_pixfmt_from_fb_varinfo(&fbdev->varinfo);

    if (pix_fmt == PIX_FMT_NONE) {

        ret = AVERROR(EINVAL);

        av_log(avctx, AV_LOG_ERROR,

               "Framebuffer pixel format not supported.\n");

        goto fail;

    }



    fbdev->width           = fbdev->varinfo.xres;

    fbdev->heigth          = fbdev->varinfo.yres;

    fbdev->bytes_per_pixel = (fbdev->varinfo.bits_per_pixel + 7) >> 3;

    fbdev->frame_linesize  = fbdev->width * fbdev->bytes_per_pixel;

    fbdev->frame_size      = fbdev->frame_linesize * fbdev->heigth;

    fbdev->time_frame      = AV_NOPTS_VALUE;

    fbdev->data = mmap(NULL, fbdev->fixinfo.smem_len, PROT_READ, MAP_SHARED, fbdev->fd, 0);

    if (fbdev->data == MAP_FAILED) {

        ret = AVERROR(errno);

        av_log(avctx, AV_LOG_ERROR, "Error in mmap(): %s\n", strerror(errno));

        goto fail;

    }



    st->codec->codec_type = AVMEDIA_TYPE_VIDEO;

    st->codec->codec_id   = CODEC_ID_RAWVIDEO;

    st->codec->width      = fbdev->width;

    st->codec->height     = fbdev->heigth;

    st->codec->pix_fmt    = pix_fmt;

    st->codec->time_base  = (AVRational){fbdev->framerate_q.den, fbdev->framerate_q.num};

    st->codec->bit_rate   =

        fbdev->width * fbdev->heigth * fbdev->bytes_per_pixel * av_q2d(fbdev->framerate_q) * 8;



    av_log(avctx, AV_LOG_INFO,

           "w:%d h:%d bpp:%d pixfmt:%s fps:%d/%d bit_rate:%d\n",

           fbdev->width, fbdev->heigth, fbdev->varinfo.bits_per_pixel,

           av_pix_fmt_descriptors[pix_fmt].name,

           fbdev->framerate_q.num, fbdev->framerate_q.den,

           st->codec->bit_rate);

    return 0;



fail:

    close(fbdev->fd);

    return ret;

}
