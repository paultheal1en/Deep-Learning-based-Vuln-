static int get_moov_size(AVFormatContext *s)

{

    int ret;

    AVIOContext *moov_buf;

    MOVMuxContext *mov = s->priv_data;



    if ((ret = ffio_open_null_buf(&moov_buf)) < 0)

        return ret;

    mov_write_moov_tag(moov_buf, mov, s);

    return ffio_close_null_buf(moov_buf);

}
