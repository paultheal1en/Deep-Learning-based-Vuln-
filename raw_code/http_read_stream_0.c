static int http_read_stream(URLContext *h, uint8_t *buf, int size)

{

    HTTPContext *s = h->priv_data;

    int err, new_location, read_ret;

    int64_t seek_ret;

    int reconnect_delay = 0;



    if (!s->hd)

        return AVERROR_EOF;



    if (s->end_chunked_post && !s->end_header) {

        err = http_read_header(h, &new_location);

        if (err < 0)

            return err;

    }



#if CONFIG_ZLIB

    if (s->compressed)

        return http_buf_read_compressed(h, buf, size);

#endif /* CONFIG_ZLIB */

    read_ret = http_buf_read(h, buf, size);

    while ((read_ret  < 0           && s->reconnect        && (!h->is_streamed || s->reconnect_streamed) && s->filesize > 0 && s->off < s->filesize)

        || (read_ret == AVERROR_EOF && s->reconnect_at_eof && (!h->is_streamed || s->reconnect_streamed))) {

        uint64_t target = h->is_streamed ? 0 : s->off;



        if (read_ret == AVERROR_EXIT)

            return read_ret;



        if (reconnect_delay > s->reconnect_delay_max)

            return AVERROR(EIO);



        av_log(h, AV_LOG_WARNING, "Will reconnect at %"PRIu64" in %d second(s), error=%s.\n", s->off, reconnect_delay, av_err2str(read_ret));

        err = ff_network_sleep_interruptible(1000U*1000*reconnect_delay, &h->interrupt_callback);

        if (err != AVERROR(ETIMEDOUT))

            return err;

        reconnect_delay = 1 + 2*reconnect_delay;

        seek_ret = http_seek_internal(h, target, SEEK_SET, 1);

        if (seek_ret >= 0 && seek_ret != target) {

            av_log(h, AV_LOG_ERROR, "Failed to reconnect at %"PRIu64".\n", target);

            return read_ret;

        }



        read_ret = http_buf_read(h, buf, size);

    }



    return read_ret;

}
