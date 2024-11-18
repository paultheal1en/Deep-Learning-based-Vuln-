static int64_t ftp_seek(URLContext *h, int64_t pos, int whence)

{

    FTPContext *s = h->priv_data;

    int err;

    int64_t new_pos, fake_pos;



    av_dlog(h, "ftp protocol seek %"PRId64" %d\n", pos, whence);



    switch(whence) {

    case AVSEEK_SIZE:

        return s->filesize;

    case SEEK_SET:

        new_pos = pos;

        break;

    case SEEK_CUR:

        new_pos = s->position + pos;

        break;

    case SEEK_END:

        if (s->filesize < 0)

            return AVERROR(EIO);

        new_pos = s->filesize + pos;

        break;

    default:

        return AVERROR(EINVAL);

    }



    if (h->is_streamed)

        return AVERROR(EIO);



    if (new_pos < 0) {

        av_log(h, AV_LOG_ERROR, "Seeking to nagative position.\n");

        return AVERROR(EINVAL);

    }



    fake_pos = s->filesize != -1 ? FFMIN(new_pos, s->filesize) : new_pos;

    if (fake_pos != s->position) {

        if ((err = ftp_abort(h)) < 0)

            return err;

        s->position = fake_pos;

    }

    return new_pos;

}
