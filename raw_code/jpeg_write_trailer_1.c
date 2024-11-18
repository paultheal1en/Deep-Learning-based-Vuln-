static int jpeg_write_trailer(AVFormatContext *s1)

{

    JpegContext *s = s1->priv_data;

    av_free(s);

    return 0;

}
