static int ftp_connect_data_connection(URLContext *h)

{

    int err;

    char buf[CONTROL_BUFFER_SIZE], opts_format[20];

    AVDictionary *opts = NULL;

    FTPContext *s = h->priv_data;



    if (!s->conn_data) {

        /* Enter passive mode */

        if ((err = ftp_passive_mode(s)) < 0) {

            av_dlog(h, "Set passive mode failed\n");

            return err;

        }

        /* Open data connection */

        ff_url_join(buf, sizeof(buf), "tcp", NULL, s->hostname, s->server_data_port, NULL);

        if (s->rw_timeout != -1) {

            snprintf(opts_format, sizeof(opts_format), "%d", s->rw_timeout);

            av_dict_set(&opts, "timeout", opts_format, 0);

        } /* if option is not given, don't pass it and let tcp use its own default */

        err = ffurl_open(&s->conn_data, buf, AVIO_FLAG_READ_WRITE,

                         &h->interrupt_callback, &opts);

        av_dict_free(&opts);

        if (err < 0)

            return err;



        if (s->position)

            if ((err = ftp_restart(s, s->position)) < 0)

                return err;

    }

    s->state = READY;

    return 0;

}
