static int gif_read_close(AVFormatContext *s1)

{

    GifState *s = s1->priv_data;

    av_free(s->image_buf);

    return 0;

}
