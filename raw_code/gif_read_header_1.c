static int gif_read_header(AVFormatContext * s1,

                           AVFormatParameters * ap)

{

    GifState *s = s1->priv_data;

    ByteIOContext *f = s1->pb;

    AVStream *st;



    s->f = f;

    if (gif_read_header1(s) < 0)

        return -1;



    /* allocate image buffer */

    s->image_linesize = s->screen_width * 3;

    s->image_buf = av_malloc(s->screen_height * s->image_linesize);

    if (!s->image_buf)

        return AVERROR(ENOMEM);

    s->pix_fmt = PIX_FMT_RGB24;

    /* now we are ready: build format streams */

    st = av_new_stream(s1, 0);

    if (!st)

        return -1;



    st->codec->codec_type = CODEC_TYPE_VIDEO;

    st->codec->codec_id = CODEC_ID_RAWVIDEO;

    st->codec->time_base.den = 5;

    st->codec->time_base.num = 1;

    /* XXX: check if screen size is always valid */

    st->codec->width = s->screen_width;

    st->codec->height = s->screen_height;

    st->codec->pix_fmt = PIX_FMT_RGB24;

    return 0;

}
