static int copy_from(IpvideoContext *s, AVFrame *src, AVFrame *dst, int delta_x, int delta_y)

{

    int current_offset = s->pixel_ptr - dst->data[0];

    int motion_offset = current_offset + delta_y * dst->linesize[0]

                       + delta_x * (1 + s->is_16bpp);

    if (motion_offset < 0) {

        av_log(s->avctx, AV_LOG_ERROR, " Interplay video: motion offset < 0 (%d)\n", motion_offset);

        return AVERROR_INVALIDDATA;

    } else if (motion_offset > s->upper_motion_limit_offset) {

        av_log(s->avctx, AV_LOG_ERROR, " Interplay video: motion offset above limit (%d >= %d)\n",

            motion_offset, s->upper_motion_limit_offset);

        return AVERROR_INVALIDDATA;

    }

    if (src->data[0] == NULL) {

        av_log(s->avctx, AV_LOG_ERROR, "Invalid decode type, corrupted header?\n");

        return AVERROR(EINVAL);

    }

    s->hdsp.put_pixels_tab[!s->is_16bpp][0](s->pixel_ptr, src->data[0] + motion_offset,

                                            dst->linesize[0], 8);

    return 0;

}
