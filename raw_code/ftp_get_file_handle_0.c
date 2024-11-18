static int ftp_get_file_handle(URLContext *h)

{

    FTPContext *s = h->priv_data;



    av_dlog(h, "ftp protocol get_file_handle\n");



    if (s->conn_data)

        return ffurl_get_file_handle(s->conn_data);



    return AVERROR(EIO);

}
