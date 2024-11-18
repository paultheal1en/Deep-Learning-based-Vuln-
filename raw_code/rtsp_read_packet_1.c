static int rtsp_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    RTSPState *rt = s->priv_data;

    int ret;

    RTSPMessageHeader reply1, *reply = &reply1;

    char cmd[1024];



    if (rt->server_type == RTSP_SERVER_REAL) {

        int i;

        enum AVDiscard cache[MAX_STREAMS];



        for (i = 0; i < s->nb_streams; i++)

            cache[i] = s->streams[i]->discard;



        if (!rt->need_subscription) {

            if (memcmp (cache, rt->real_setup_cache,

                        sizeof(enum AVDiscard) * s->nb_streams)) {

                av_strlcatf(cmd, sizeof(cmd),

                            "SET_PARAMETER %s RTSP/1.0\r\n"

                            "Unsubscribe: %s\r\n",

                            s->filename, rt->last_subscription);

                rtsp_send_cmd(s, cmd, reply, NULL);

                if (reply->status_code != RTSP_STATUS_OK)

                    return AVERROR_INVALIDDATA;

                rt->need_subscription = 1;

            }

        }



        if (rt->need_subscription) {

            int r, rule_nr, first = 1;



            memcpy(rt->real_setup_cache, cache,

                   sizeof(enum AVDiscard) * s->nb_streams);

            rt->last_subscription[0] = 0;



            snprintf(cmd, sizeof(cmd),

                     "SET_PARAMETER %s RTSP/1.0\r\n"

                     "Subscribe: ",

                     s->filename);

            for (i = 0; i < rt->nb_rtsp_streams; i++) {

                rule_nr = 0;

                for (r = 0; r < s->nb_streams; r++) {

                    if (s->streams[r]->priv_data == rt->rtsp_streams[i]) {

                        if (s->streams[r]->discard != AVDISCARD_ALL) {

                            if (!first)

                                av_strlcat(rt->last_subscription, ",",

                                           sizeof(rt->last_subscription));

                            ff_rdt_subscribe_rule(

                                rt->last_subscription,

                                sizeof(rt->last_subscription), i, rule_nr);

                            first = 0;

                        }

                        rule_nr++;

                    }

                }

            }

            av_strlcatf(cmd, sizeof(cmd), "%s\r\n", rt->last_subscription);

            rtsp_send_cmd(s, cmd, reply, NULL);

            if (reply->status_code != RTSP_STATUS_OK)

                return AVERROR_INVALIDDATA;

            rt->need_subscription = 0;



            if (rt->state == RTSP_STATE_PLAYING)

                rtsp_read_play (s);

        }

    }



    ret = rtsp_fetch_packet(s, pkt);

    if (ret < 0)

        return ret;



    /* send dummy request to keep TCP connection alive */

    if ((rt->server_type == RTSP_SERVER_WMS ||

         rt->server_type == RTSP_SERVER_REAL) &&

        (av_gettime() - rt->last_cmd_time) / 1000000 >= rt->timeout / 2) {

        if (rt->server_type == RTSP_SERVER_WMS) {

            snprintf(cmd, sizeof(cmd) - 1,

                     "GET_PARAMETER %s RTSP/1.0\r\n",

                     s->filename);

            rtsp_send_cmd_async(s, cmd, reply, NULL);

        } else {

            rtsp_send_cmd_async(s, "OPTIONS * RTSP/1.0\r\n",

                                reply, NULL);

        }

    }



    return 0;

}
