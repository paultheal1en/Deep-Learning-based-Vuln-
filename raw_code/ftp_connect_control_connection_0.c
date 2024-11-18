static int ftp_connect_control_connection(URLContext *h)

{

    char buf[CONTROL_BUFFER_SIZE], opts_format[20];

    int err;

    AVDictionary *opts = NULL;

    FTPContext *s = h->priv_data;

    const int connect_codes[] = {220, 0};



    s->conn_control_block_flag = 0;



    if (!s->conn_control) {

        ff_url_join(buf, sizeof(buf), "tcp", NULL,

                    s->hostname, s->server_control_port, NULL);

        if (s->rw_timeout != -1) {

            snprintf(opts_format, sizeof(opts_format), "%d", s->rw_timeout);

            av_dict_set(&opts, "timeout", opts_format, 0);

        } /* if option is not given, don't pass it and let tcp use its own default */

        err = ffurl_open(&s->conn_control, buf, AVIO_FLAG_READ_WRITE,

                         &s->conn_control_interrupt_cb, &opts);

        av_dict_free(&opts);

        if (err < 0) {

            av_log(h, AV_LOG_ERROR, "Cannot open control connection\n");

            return err;

        }



        /* consume all messages from server */

        if (!ftp_status(s, NULL, connect_codes)) {

            av_log(h, AV_LOG_ERROR, "FTP server not ready for new users\n");

            err = AVERROR(EACCES);

            return err;

        }



        if ((err = ftp_auth(s)) < 0) {

            av_log(h, AV_LOG_ERROR, "FTP authentication failed\n");

            return err;

        }



        if ((err = ftp_type(s)) < 0) {

            av_dlog(h, "Set content type failed\n");

            return err;

        }

    }

    return 0;

}
