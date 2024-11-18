static int rtsp_read_packet(AVFormatContext *s,

                            AVPacket *pkt)

{

    RTSPState *rt = s->priv_data;

    RTSPStream *rtsp_st;

    int ret, len;

    uint8_t buf[RTP_MAX_PACKET_LENGTH];



    /* get next frames from the same RTP packet */

    if (rt->cur_rtp) {

        ret = rtp_parse_packet(rt->cur_rtp, pkt, NULL, 0);

        if (ret == 0) {

            rt->cur_rtp = NULL;

            return 0;

        } else if (ret == 1) {

            return 0;

        } else {

            rt->cur_rtp = NULL;

        }

    }



    /* read next RTP packet */

 redo:

    switch(rt->protocol) {

    default:

    case RTSP_PROTOCOL_RTP_TCP:

        len = tcp_read_packet(s, &rtsp_st, buf, sizeof(buf));

        break;

    case RTSP_PROTOCOL_RTP_UDP:

    case RTSP_PROTOCOL_RTP_UDP_MULTICAST:

        len = udp_read_packet(s, &rtsp_st, buf, sizeof(buf));

        if (rtsp_st->rtp_ctx)

            rtp_check_and_send_back_rr(rtsp_st->rtp_ctx, len);

        break;

    }

    if (len < 0)

        return AVERROR_IO;

    ret = rtp_parse_packet(rtsp_st->rtp_ctx, pkt, buf, len);

    if (ret < 0)

        goto redo;

    if (ret == 1) {

        /* more packets may follow, so we save the RTP context */

        rt->cur_rtp = rtsp_st->rtp_ctx;

    }

    return 0;

}
