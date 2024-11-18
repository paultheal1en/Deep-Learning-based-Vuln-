static int ftp_write(URLContext *h, const unsigned char *buf, int size)

{

    int err;

    FTPContext *s = h->priv_data;

    int written;



    av_dlog(h, "ftp protocol write %d bytes\n", size);



    if (s->state == DISCONNECTED) {

        if ((err = ftp_connect_data_connection(h)) < 0)

            return err;

    }

    if (s->state == READY) {

        if ((err = ftp_store(s)) < 0)

            return err;

    }

    if (s->conn_data && s->state == UPLOADING) {

        written = ffurl_write(s->conn_data, buf, size);

        if (written > 0) {

            s->position += written;

            s->filesize = FFMAX(s->filesize, s->position);

        }

        return written;

    }



    av_log(h, AV_LOG_ERROR, "FTP write failed\n");

    return AVERROR(EIO);

}
