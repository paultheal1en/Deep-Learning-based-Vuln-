static int mpegts_read_header(AVFormatContext *s,

                              AVFormatParameters *ap)

{

    MpegTSContext *ts = s->priv_data;

    AVIOContext *pb = s->pb;

    uint8_t buf[5*1024];

    int len;

    int64_t pos;



#if FF_API_FORMAT_PARAMETERS

    if (ap) {

        if (ap->mpeg2ts_compute_pcr)

            ts->mpeg2ts_compute_pcr = ap->mpeg2ts_compute_pcr;

        if(ap->mpeg2ts_raw){

            av_log(s, AV_LOG_ERROR, "use mpegtsraw_demuxer!\n");

            return -1;

        }

    }

#endif



    /* read the first 1024 bytes to get packet size */

    pos = avio_tell(pb);

    len = avio_read(pb, buf, sizeof(buf));

    if (len != sizeof(buf))

        goto fail;

    ts->raw_packet_size = get_packet_size(buf, sizeof(buf));

    if (ts->raw_packet_size <= 0)

        goto fail;

    ts->stream = s;

    ts->auto_guess = 0;



    if (s->iformat == &ff_mpegts_demuxer) {

        /* normal demux */



        /* first do a scaning to get all the services */

        if (avio_seek(pb, pos, SEEK_SET) < 0)

            av_log(s, AV_LOG_ERROR, "Unable to seek back to the start\n");



        mpegts_open_section_filter(ts, SDT_PID, sdt_cb, ts, 1);



        mpegts_open_section_filter(ts, PAT_PID, pat_cb, ts, 1);



        handle_packets(ts, s->probesize / ts->raw_packet_size);

        /* if could not find service, enable auto_guess */



        ts->auto_guess = 1;



        av_dlog(ts->stream, "tuning done\n");



        s->ctx_flags |= AVFMTCTX_NOHEADER;

    } else {

        AVStream *st;

        int pcr_pid, pid, nb_packets, nb_pcrs, ret, pcr_l;

        int64_t pcrs[2], pcr_h;

        int packet_count[2];

        uint8_t packet[TS_PACKET_SIZE];



        /* only read packets */



        st = av_new_stream(s, 0);

        if (!st)

            goto fail;

        av_set_pts_info(st, 60, 1, 27000000);

        st->codec->codec_type = AVMEDIA_TYPE_DATA;

        st->codec->codec_id = CODEC_ID_MPEG2TS;



        /* we iterate until we find two PCRs to estimate the bitrate */

        pcr_pid = -1;

        nb_pcrs = 0;

        nb_packets = 0;

        for(;;) {

            ret = read_packet(s, packet, ts->raw_packet_size);

            if (ret < 0)

                return -1;

            pid = AV_RB16(packet + 1) & 0x1fff;

            if ((pcr_pid == -1 || pcr_pid == pid) &&

                parse_pcr(&pcr_h, &pcr_l, packet) == 0) {

                pcr_pid = pid;

                packet_count[nb_pcrs] = nb_packets;

                pcrs[nb_pcrs] = pcr_h * 300 + pcr_l;

                nb_pcrs++;

                if (nb_pcrs >= 2)

                    break;

            }

            nb_packets++;

        }



        /* NOTE1: the bitrate is computed without the FEC */

        /* NOTE2: it is only the bitrate of the start of the stream */

        ts->pcr_incr = (pcrs[1] - pcrs[0]) / (packet_count[1] - packet_count[0]);

        ts->cur_pcr = pcrs[0] - ts->pcr_incr * packet_count[0];

        s->bit_rate = (TS_PACKET_SIZE * 8) * 27e6 / ts->pcr_incr;

        st->codec->bit_rate = s->bit_rate;

        st->start_time = ts->cur_pcr;

        av_dlog(ts->stream, "start=%0.3f pcr=%0.3f incr=%d\n",

                st->start_time / 1000000.0, pcrs[0] / 27e6, ts->pcr_incr);

    }



    avio_seek(pb, pos, SEEK_SET);

    return 0;

 fail:

    return -1;

}
