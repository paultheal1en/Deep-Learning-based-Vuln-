static void handle_stream_probing(AVStream *st)

{

    if (st->codec->codec_id == AV_CODEC_ID_PCM_S16LE) {

        st->request_probe = AVPROBE_SCORE_EXTENSION;

        st->probe_packets = FFMIN(st->probe_packets, 14);

    }

}
