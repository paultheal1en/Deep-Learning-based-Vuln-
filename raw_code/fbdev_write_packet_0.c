static int fbdev_write_packet(AVFormatContext *h, AVPacket *pkt)

{

    FBDevContext *fbdev = h->priv_data;

    uint8_t *pin, *pout;

    enum AVPixelFormat fb_pix_fmt;

    int disp_height;

    int bytes_to_copy;

    AVCodecContext *codec_ctx = h->streams[fbdev->index]->codec;

    enum AVPixelFormat video_pix_fmt = codec_ctx->pix_fmt;

    int video_width = codec_ctx->width;

    int video_height = codec_ctx->height;

    int bytes_per_pixel = ((codec_ctx->bits_per_coded_sample + 7) >> 3);

    int src_line_size = video_width * bytes_per_pixel;

    int i;



    if (fbdev->index != pkt->stream_index)

        return 0;



    if (ioctl(fbdev->fd, FBIOGET_VSCREENINFO, &fbdev->varinfo) < 0)

        av_log(h, AV_LOG_WARNING,

               "Error refreshing variable info: %s\n", av_err2str(AVERROR(errno)));



    fb_pix_fmt = ff_get_pixfmt_from_fb_varinfo(&fbdev->varinfo);



    if (fb_pix_fmt != video_pix_fmt) {

        av_log(h, AV_LOG_ERROR, "Pixel format %s is not supported, use %s\n",

               av_get_pix_fmt_name(video_pix_fmt), av_get_pix_fmt_name(fb_pix_fmt));

        return AVERROR(EINVAL);

    }



    disp_height = FFMIN(fbdev->varinfo.yres, video_height);

    bytes_to_copy = FFMIN(fbdev->varinfo.xres, video_width) * bytes_per_pixel;



    pin  = pkt->data;

    pout = fbdev->data +

           bytes_per_pixel * fbdev->varinfo.xoffset +

           fbdev->varinfo.yoffset * fbdev->fixinfo.line_length;



    if (fbdev->xoffset) {

        if (fbdev->xoffset < 0) {

            if (-fbdev->xoffset >= video_width) //nothing to display

                return 0;

            bytes_to_copy += fbdev->xoffset * bytes_per_pixel;

            pin -= fbdev->xoffset * bytes_per_pixel;

        } else {

            int diff = (video_width + fbdev->xoffset) - fbdev->varinfo.xres;

            if (diff > 0) {

                if (diff >= video_width) //nothing to display

                    return 0;

                bytes_to_copy -= diff * bytes_per_pixel;

            }

            pout += bytes_per_pixel * fbdev->xoffset;

        }

    }



    if (fbdev->yoffset) {

        if (fbdev->yoffset < 0) {

            if (-fbdev->yoffset >= video_height) //nothing to display

                return 0;

            disp_height += fbdev->yoffset;

            pin -= fbdev->yoffset * src_line_size;

        } else {

            int diff = (video_height + fbdev->yoffset) - fbdev->varinfo.yres;

            if (diff > 0) {

                if (diff >= video_height) //nothing to display

                    return 0;

                disp_height -= diff;

            }

            pout += fbdev->yoffset * fbdev->fixinfo.line_length;

        }

    }



    for (i = 0; i < disp_height; i++) {

        memcpy(pout, pin, bytes_to_copy);

        pout += fbdev->fixinfo.line_length;

        pin  += src_line_size;

    }



    return 0;

}
