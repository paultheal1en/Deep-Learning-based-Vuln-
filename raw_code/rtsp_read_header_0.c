static int rtsp_read_header(AVFormatContext *s,

                            AVFormatParameters *ap)

{

    RTSPState *rt = s->priv_data;

    char host[1024], path[1024], tcpname[1024], cmd[2048];

    URLContext *rtsp_hd;

    int port, i, ret, err;

    RTSPHeader reply1, *reply = &reply1;

    unsigned char *content = NULL;

    AVStream *st;

    RTSPStream *rtsp_st;

    int protocol_mask;



    rtsp_abort_req = 0;

    

    /* extract hostname and port */

    url_split(NULL, 0,

              host, sizeof(host), &port, path, sizeof(path), s->filename);

    if (port < 0)

        port = RTSP_DEFAULT_PORT;



    /* open the tcp connexion */

    snprintf(tcpname, sizeof(tcpname), "tcp://%s:%d", host, port);

    if (url_open(&rtsp_hd, tcpname, URL_RDWR) < 0)

        return AVERROR_IO;

    rt->rtsp_hd = rtsp_hd;

    rt->seq = 0;

    

    /* describe the stream */

    snprintf(cmd, sizeof(cmd), 

             "DESCRIBE %s RTSP/1.0\r\n"

             "Accept: application/sdp\r\n",

             s->filename);

    rtsp_send_cmd(s, cmd, reply, &content);

    if (!content) {

        err = AVERROR_INVALIDDATA;

        goto fail;

    }

    if (reply->status_code != RTSP_STATUS_OK) {

        err = AVERROR_INVALIDDATA;

        goto fail;

    }

        

    /* now we got the SDP description, we parse it */

    ret = sdp_parse(s, (const char *)content);

    av_freep(&content);

    if (ret < 0) {

        err = AVERROR_INVALIDDATA;

        goto fail;

    }

    

    protocol_mask = rtsp_default_protocols;



    /* for each stream, make the setup request */

    /* XXX: we assume the same server is used for the control of each

       RTSP stream */

    for(i=0;i<s->nb_streams;i++) {

        char transport[2048];

        AVInputFormat *fmt;



        st = s->streams[i];

        rtsp_st = st->priv_data;



        /* compute available transports */

        transport[0] = '\0';



        /* RTP/UDP */

        if (protocol_mask & (1 << RTSP_PROTOCOL_RTP_UDP)) {

            char buf[256];

            int j;



            /* first try in specified port range */

            if (rtsp_rtp_port_min != 0) {

                for(j=rtsp_rtp_port_min;j<=rtsp_rtp_port_max;j++) {

                    snprintf(buf, sizeof(buf), "rtp://?localport=%d", j);

                    if (!av_open_input_file(&rtsp_st->ic, buf, 

                                            &rtp_demux, 0, NULL))

                        goto rtp_opened;

                }

            }



            /* then try on any port */

            if (av_open_input_file(&rtsp_st->ic, "rtp://", 

                                       &rtp_demux, 0, NULL) < 0) {

                    err = AVERROR_INVALIDDATA;

                    goto fail;

            }



        rtp_opened:

            port = rtp_get_local_port(url_fileno(&rtsp_st->ic->pb));

            if (transport[0] != '\0')

                pstrcat(transport, sizeof(transport), ",");

            snprintf(transport + strlen(transport), sizeof(transport) - strlen(transport) - 1,

                     "RTP/AVP/UDP;unicast;client_port=%d-%d",

                     port, port + 1);

        }



        /* RTP/TCP */

        if (protocol_mask & (1 << RTSP_PROTOCOL_RTP_TCP)) {

            if (transport[0] != '\0')

                pstrcat(transport, sizeof(transport), ",");

            snprintf(transport + strlen(transport), sizeof(transport) - strlen(transport) - 1,

                     "RTP/AVP/TCP");

        }



        if (protocol_mask & (1 << RTSP_PROTOCOL_RTP_UDP_MULTICAST)) {

            if (transport[0] != '\0')

                pstrcat(transport, sizeof(transport), ",");

            snprintf(transport + strlen(transport), 

                     sizeof(transport) - strlen(transport) - 1,

                     "RTP/AVP/UDP;multicast");

        }

        snprintf(cmd, sizeof(cmd), 

                 "SETUP %s RTSP/1.0\r\n"

                 "Transport: %s\r\n",

                 rtsp_st->control_url, transport);

        rtsp_send_cmd(s, cmd, reply, NULL);

        if (reply->status_code != RTSP_STATUS_OK ||

            reply->nb_transports != 1) {

            err = AVERROR_INVALIDDATA;

            goto fail;

        }



        /* XXX: same protocol for all streams is required */

        if (i > 0) {

            if (reply->transports[0].protocol != rt->protocol) {

                err = AVERROR_INVALIDDATA;

                goto fail;

            }

        } else {

            rt->protocol = reply->transports[0].protocol;

        }



        /* close RTP connection if not choosen */

        if (reply->transports[0].protocol != RTSP_PROTOCOL_RTP_UDP &&

            (protocol_mask & (1 << RTSP_PROTOCOL_RTP_UDP))) {

            av_close_input_file(rtsp_st->ic);

            rtsp_st->ic = NULL;

        }



        switch(reply->transports[0].protocol) {

        case RTSP_PROTOCOL_RTP_TCP:

            fmt = &rtp_demux;

            if (av_open_input_file(&rtsp_st->ic, "null", fmt, 0, NULL) < 0) {

                err = AVERROR_INVALIDDATA;

                goto fail;

            }

            rtsp_st->interleaved_min = reply->transports[0].interleaved_min;

            rtsp_st->interleaved_max = reply->transports[0].interleaved_max;

            break;

            

        case RTSP_PROTOCOL_RTP_UDP:

            {

                char url[1024];

                

                /* XXX: also use address if specified */

                snprintf(url, sizeof(url), "rtp://%s:%d", 

                         host, reply->transports[0].server_port_min);

                if (rtp_set_remote_url(url_fileno(&rtsp_st->ic->pb), url) < 0) {

                    err = AVERROR_INVALIDDATA;

                    goto fail;

                }

            }

            break;

        case RTSP_PROTOCOL_RTP_UDP_MULTICAST:

            {

                char url[1024];

                int ttl;



                fmt = &rtp_demux;

                ttl = reply->transports[0].ttl;

                if (!ttl)

                    ttl = 16;

                snprintf(url, sizeof(url), "rtp://%s:%d?multicast=1&ttl=%d", 

                         host, 

                         reply->transports[0].server_port_min,

                         ttl);

                if (av_open_input_file(&rtsp_st->ic, url, fmt, 0, NULL) < 0) {

                    err = AVERROR_INVALIDDATA;

                    goto fail;

                }

            }

            break;

        }

    }



    /* use callback if available to extend setup */

    if (ff_rtsp_callback) {

        if (ff_rtsp_callback(RTSP_ACTION_CLIENT_SETUP, rt->session_id, 

                             NULL, 0, rt->last_reply) < 0) {

            err = AVERROR_INVALIDDATA;

            goto fail;

        }

    }

                         

    /* start playing */

    snprintf(cmd, sizeof(cmd), 

             "PLAY %s RTSP/1.0\r\n"

             "Range: npt=0-\r\n",

             s->filename);

    rtsp_send_cmd(s, cmd, reply, NULL);

    if (reply->status_code != RTSP_STATUS_OK) {

        err = AVERROR_INVALIDDATA;

        goto fail;

    }



#if 0

    /* open TCP with bufferized input */

    if (rt->protocol == RTSP_PROTOCOL_RTP_TCP) {

        if (url_fdopen(&rt->rtsp_gb, rt->rtsp_hd) < 0) {

            err = AVERROR_NOMEM;

            goto fail;

        }

    }

#endif



    return 0;

 fail:

    for(i=0;i<s->nb_streams;i++) {

        st = s->streams[i];

        rtsp_st = st->priv_data;

        if (rtsp_st) {

            if (rtsp_st->ic)

                av_close_input_file(rtsp_st->ic);

        }

        av_free(rtsp_st);

    }

    av_freep(&content);

    url_close(rt->rtsp_hd);

    return err;

}
