static void gen_connect(URLContext *s, RTMPContext *rt)

{

    RTMPPacket pkt;

    uint8_t ver[64], *p;



    ff_rtmp_packet_create(&pkt, RTMP_SYSTEM_CHANNEL, RTMP_PT_INVOKE, 0, 4096);

    p = pkt.data;



    ff_amf_write_string(&p, "connect");

    ff_amf_write_number(&p, ++rt->nb_invokes);

    ff_amf_write_object_start(&p);

    ff_amf_write_field_name(&p, "app");

    ff_amf_write_string(&p, rt->app);



    if (rt->is_input) {

        snprintf(ver, sizeof(ver), "%s %d,%d,%d,%d", RTMP_CLIENT_PLATFORM, RTMP_CLIENT_VER1,

                 RTMP_CLIENT_VER2, RTMP_CLIENT_VER3, RTMP_CLIENT_VER4);

    } else {

        snprintf(ver, sizeof(ver), "FMLE/3.0 (compatible; %s)", LIBAVFORMAT_IDENT);

        ff_amf_write_field_name(&p, "type");

        ff_amf_write_string(&p, "nonprivate");

    }

    ff_amf_write_field_name(&p, "flashVer");

    ff_amf_write_string(&p, ver);

    ff_amf_write_field_name(&p, "tcUrl");

    ff_amf_write_string(&p, rt->tcurl);

    if (rt->is_input) {

        ff_amf_write_field_name(&p, "fpad");

        ff_amf_write_bool(&p, 0);

        ff_amf_write_field_name(&p, "capabilities");

        ff_amf_write_number(&p, 15.0);



        /* Tell the server we support all the audio codecs except

         * SUPPORT_SND_INTEL (0x0008) and SUPPORT_SND_UNUSED (0x0010)

         * which are unused in the RTMP protocol implementation. */

        ff_amf_write_field_name(&p, "audioCodecs");

        ff_amf_write_number(&p, 4071.0);

        ff_amf_write_field_name(&p, "videoCodecs");

        ff_amf_write_number(&p, 252.0);

        ff_amf_write_field_name(&p, "videoFunction");

        ff_amf_write_number(&p, 1.0);

    }

    ff_amf_write_object_end(&p);



    pkt.data_size = p - pkt.data;



    ff_rtmp_packet_write(rt->stream, &pkt, rt->chunk_size, rt->prev_pkt[1]);

    ff_rtmp_packet_destroy(&pkt);

}
