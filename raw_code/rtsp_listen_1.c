static int rtsp_listen(AVFormatContext *s)

{

    RTSPState *rt = s->priv_data;

    char proto[128], host[128], path[512], auth[128];

    char uri[500];

    int port;

    int default_port = RTSP_DEFAULT_PORT;

    char tcpname[500];

    const char *lower_proto = "tcp";

    unsigned char rbuf[4096];

    unsigned char method[10];

    int rbuflen = 0;

    int ret;

    enum RTSPMethod methodcode;



    if (!rt->protocols) {

        rt->protocols = ffurl_get_protocols(NULL, NULL);

        if (!rt->protocols)

            return AVERROR(ENOMEM);

    }



    /* extract hostname and port */

    av_url_split(proto, sizeof(proto), auth, sizeof(auth), host, sizeof(host),

                 &port, path, sizeof(path), s->filename);



    /* ff_url_join. No authorization by now (NULL) */

    ff_url_join(rt->control_uri, sizeof(rt->control_uri), proto, NULL, host,

                port, "%s", path);



    if (!strcmp(proto, "rtsps")) {

        lower_proto  = "tls";

        default_port = RTSPS_DEFAULT_PORT;

    }



    if (port < 0)

        port = default_port;



    /* Create TCP connection */

    ff_url_join(tcpname, sizeof(tcpname), lower_proto, NULL, host, port,

                "?listen&listen_timeout=%d", rt->initial_timeout * 1000);



    if (ret = ffurl_open(&rt->rtsp_hd, tcpname, AVIO_FLAG_READ_WRITE,

                         &s->interrupt_callback, NULL, rt->protocols)) {

        av_log(s, AV_LOG_ERROR, "Unable to open RTSP for listening\n");

        return ret;

    }

    rt->state       = RTSP_STATE_IDLE;

    rt->rtsp_hd_out = rt->rtsp_hd;

    for (;;) { /* Wait for incoming RTSP messages */

        ret = read_line(s, rbuf, sizeof(rbuf), &rbuflen);

        if (ret < 0)

            return ret;

        ret = parse_command_line(s, rbuf, rbuflen, uri, sizeof(uri), method,

                                 sizeof(method), &methodcode);

        if (ret) {

            av_log(s, AV_LOG_ERROR, "RTSP: Unexpected Command\n");

            return ret;

        }



        if (methodcode == ANNOUNCE) {

            ret       = rtsp_read_announce(s);

            rt->state = RTSP_STATE_PAUSED;

        } else if (methodcode == OPTIONS) {

            ret = rtsp_read_options(s);

        } else if (methodcode == RECORD) {

            ret = rtsp_read_record(s);

            if (!ret)

                return 0; // We are ready for streaming

        } else if (methodcode == SETUP)

            ret = rtsp_read_setup(s, host, uri);

        if (ret) {

            ffurl_close(rt->rtsp_hd);

            return AVERROR_INVALIDDATA;

        }

    }

    return 0;

}
