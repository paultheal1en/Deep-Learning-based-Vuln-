static int ftp_shutdown(URLContext *h, int flags)

{

    FTPContext *s = h->priv_data;



    av_dlog(h, "ftp protocol shutdown\n");



    if (s->conn_data)

        return ffurl_shutdown(s->conn_data, flags);



    return AVERROR(EIO);

}
