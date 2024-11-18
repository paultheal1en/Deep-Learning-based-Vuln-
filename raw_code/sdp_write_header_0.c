static void sdp_write_header(char *buff, int size, struct sdp_session_level *s)

{

    av_strlcatf(buff, size, "v=%d\r\n"

                            "o=- %d %d IN IPV4 %s\r\n"

                            "t=%d %d\r\n"

                            "s=%s\r\n"

                            "a=tool:libavformat " AV_STRINGIFY(LIBAVFORMAT_VERSION) "\r\n",

                            s->sdp_version,

                            s->id, s->version, s->src_addr,

                            s->start_time, s->end_time,

                            s->name[0] ? s->name : "No Name");

    dest_write(buff, size, s->dst_addr, s->ttl);

}
