static void rtcp_send_sr(AVFormatContext *s1, int64_t ntp_time)

{

    RTPDemuxContext *s = s1->priv_data;

    uint32_t rtp_ts;



#if defined(DEBUG)

    printf("RTCP: %02x %"PRIx64" %x\n", s->payload_type, ntp_time, s->timestamp);

#endif



    if (s->first_rtcp_ntp_time == AV_NOPTS_VALUE) s->first_rtcp_ntp_time = ntp_time;

    s->last_rtcp_ntp_time = ntp_time;

    rtp_ts = av_rescale_q(ntp_time - s->first_rtcp_ntp_time, AV_TIME_BASE_Q,

                          s1->streams[0]->time_base) + s->base_timestamp;

    put_byte(s1->pb, (RTP_VERSION << 6));

    put_byte(s1->pb, 200);

    put_be16(s1->pb, 6); /* length in words - 1 */

    put_be32(s1->pb, s->ssrc);

    put_be32(s1->pb, ntp_time / 1000000);

    put_be32(s1->pb, ((ntp_time % 1000000) << 32) / 1000000);

    put_be32(s1->pb, rtp_ts);

    put_be32(s1->pb, s->packet_count);

    put_be32(s1->pb, s->octet_count);

    put_flush_packet(s1->pb);

}
