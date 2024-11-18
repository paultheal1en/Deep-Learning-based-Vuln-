static int rtcp_parse_packet(RTPDemuxContext *s, const unsigned char *buf,

                             int len)

{

    int payload_len;

    while (len >= 4) {

        payload_len = FFMIN(len, (AV_RB16(buf + 2) + 1) * 4);



        switch (buf[1]) {

        case RTCP_SR:

            if (payload_len < 20) {

                av_log(NULL, AV_LOG_ERROR,

                       "Invalid length for RTCP SR packet\n");

                return AVERROR_INVALIDDATA;

            }



            s->last_rtcp_reception_time = av_gettime_relative();

            s->last_rtcp_ntp_time  = AV_RB64(buf + 8);

            s->last_rtcp_timestamp = AV_RB32(buf + 16);

            if (s->first_rtcp_ntp_time == AV_NOPTS_VALUE) {

                s->first_rtcp_ntp_time = s->last_rtcp_ntp_time;

                if (!s->base_timestamp)

                    s->base_timestamp = s->last_rtcp_timestamp;

                s->rtcp_ts_offset = s->last_rtcp_timestamp - s->base_timestamp;

            }



            break;

        case RTCP_BYE:

            return -RTCP_BYE;

        }



        buf += payload_len;

        len -= payload_len;

    }

    return -1;

}
