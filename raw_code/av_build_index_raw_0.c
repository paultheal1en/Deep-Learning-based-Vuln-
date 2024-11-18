static void av_build_index_raw(AVFormatContext *s)

{

    AVPacket pkt1, *pkt = &pkt1;

    int ret;

    AVStream *st;



    st = s->streams[0];

    av_read_frame_flush(s);

    url_fseek(&s->pb, s->data_offset, SEEK_SET);



    for(;;) {

        ret = av_read_frame(s, pkt);

        if (ret < 0)

            break;

        if (pkt->stream_index == 0 && st->parser &&

            (pkt->flags & PKT_FLAG_KEY)) {

            add_index_entry(st, st->parser->frame_offset, pkt->dts, 

                            AVINDEX_KEYFRAME);

        }

        av_free_packet(pkt);

    }

}
