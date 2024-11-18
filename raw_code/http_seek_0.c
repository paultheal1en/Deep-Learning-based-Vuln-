static int64_t http_seek(URLContext *h, int64_t off, int whence)

{

    HTTPContext *s = h->priv_data;

    URLContext *old_hd = s->hd;

    int64_t old_off = s->off;

    uint8_t old_buf[BUFFER_SIZE];

    int old_buf_size;

    AVDictionary *options = NULL;



    if (whence == AVSEEK_SIZE)

        return s->filesize;

    else if ((whence == SEEK_CUR && off == 0) || (whence == SEEK_SET && off == s->off))

        return s->off;

    else if ((s->filesize == -1 && whence == SEEK_END) || h->is_streamed)

        return -1;



    /* we save the old context in case the seek fails */

    old_buf_size = s->buf_end - s->buf_ptr;

    memcpy(old_buf, s->buf_ptr, old_buf_size);

    s->hd = NULL;

    if (whence == SEEK_CUR)

        off += s->off;

    else if (whence == SEEK_END)

        off += s->filesize;

    s->off = off;



    /* if it fails, continue on old connection */

    av_dict_copy(&options, s->chained_options, 0);

    if (http_open_cnx(h, &options) < 0) {

        av_dict_free(&options);

        memcpy(s->buffer, old_buf, old_buf_size);

        s->buf_ptr = s->buffer;

        s->buf_end = s->buffer + old_buf_size;

        s->hd = old_hd;

        s->off = old_off;

        return -1;

    }

    av_dict_free(&options);

    ffurl_close(old_hd);

    return off;

}
