static int seg_write_packet(AVFormatContext *s, AVPacket *pkt)

{

    SegmentContext *seg = s->priv_data;

    AVFormatContext *oc = seg->avf;

    AVStream *st = s->streams[pkt->stream_index];

    int64_t end_pts = seg->recording_time * seg->number;

    int ret, can_split = 1;



    if (!oc)

        return AVERROR(EINVAL);



    if (seg->has_video) {

        can_split = st->codec->codec_type == AVMEDIA_TYPE_VIDEO &&

                    pkt->flags & AV_PKT_FLAG_KEY;

    }



    if (can_split && av_compare_ts(pkt->pts, st->time_base, end_pts,

                                   AV_TIME_BASE_Q) >= 0) {

        av_log(s, AV_LOG_DEBUG, "Next segment starts at %d %"PRId64"\n",

               pkt->stream_index, pkt->pts);



        ret = segment_end(oc, seg->individual_header_trailer);



        if (!ret)

            ret = segment_start(s, seg->individual_header_trailer);



        if (ret)

            goto fail;



        oc = seg->avf;



        if (seg->list) {

            if (seg->list_type == LIST_HLS) {

                if ((ret = segment_hls_window(s, 0)) < 0)

                    goto fail;

            } else {

                avio_printf(seg->pb, "%s\n", oc->filename);

                avio_flush(seg->pb);

                if (seg->size && !(seg->number % seg->size)) {

                    avio_closep(&seg->pb);

                    if ((ret = avio_open2(&seg->pb, seg->list, AVIO_FLAG_WRITE,

                                          &s->interrupt_callback, NULL)) < 0)

                        goto fail;

                }

            }

        }

    }



    ret = ff_write_chained(oc, pkt->stream_index, pkt, s);



fail:

    if (ret < 0)

        seg_free_context(seg);



    return ret;

}
