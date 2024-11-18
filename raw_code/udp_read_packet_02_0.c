static int udp_read_packet(AVFormatContext *s, RTSPStream **prtsp_st,

                           uint8_t *buf, int buf_size)

{

    RTSPState *rt = s->priv_data;

    RTSPStream *rtsp_st;

    fd_set rfds;

    int fd, fd_max, n, i, ret, tcp_fd, timeout_cnt = 0;

    struct timeval tv;



    for (;;) {

        if (url_interrupt_cb())

            return AVERROR(EINTR);

        FD_ZERO(&rfds);

        if (rt->rtsp_hd) {

            tcp_fd = fd_max = url_get_file_handle(rt->rtsp_hd);

            FD_SET(tcp_fd, &rfds);

        } else {

            fd_max = 0;

            tcp_fd = -1;

        }

        for (i = 0; i < rt->nb_rtsp_streams; i++) {

            rtsp_st = rt->rtsp_streams[i];

            if (rtsp_st->rtp_handle) {

                /* currently, we cannot probe RTCP handle because of

                 * blocking restrictions */

                fd = url_get_file_handle(rtsp_st->rtp_handle);

                if (fd > fd_max)

                    fd_max = fd;

                FD_SET(fd, &rfds);

            }

        }

        tv.tv_sec = 0;

        tv.tv_usec = SELECT_TIMEOUT_MS * 1000;

        n = select(fd_max + 1, &rfds, NULL, NULL, &tv);

        if (n > 0) {

            timeout_cnt = 0;

            for (i = 0; i < rt->nb_rtsp_streams; i++) {

                rtsp_st = rt->rtsp_streams[i];

                if (rtsp_st->rtp_handle) {

                    fd = url_get_file_handle(rtsp_st->rtp_handle);

                    if (FD_ISSET(fd, &rfds)) {

                        ret = url_read(rtsp_st->rtp_handle, buf, buf_size);

                        if (ret > 0) {

                            *prtsp_st = rtsp_st;

                            return ret;

                        }

                    }

                }

            }

#if CONFIG_RTSP_DEMUXER

            if (tcp_fd != -1 && FD_ISSET(tcp_fd, &rfds)) {

                RTSPMessageHeader reply;



                ret = ff_rtsp_read_reply(s, &reply, NULL, 0);

                if (ret < 0)

                    return ret;

                /* XXX: parse message */

                if (rt->state != RTSP_STATE_STREAMING)

                    return 0;

            }

#endif

        } else if (n == 0 && ++timeout_cnt >= MAX_TIMEOUTS) {

            return FF_NETERROR(ETIMEDOUT);

        } else if (n < 0 && errno != EINTR)

            return AVERROR(errno);

    }

}
