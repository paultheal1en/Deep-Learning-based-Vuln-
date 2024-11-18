static int seg_check_bitstream(struct AVFormatContext *s, const AVPacket *pkt)

{

    SegmentContext *seg = s->priv_data;

    AVFormatContext *oc = seg->avf;

    if (oc->oformat->check_bitstream) {

        int ret = oc->oformat->check_bitstream(oc, pkt);

        if (ret == 1) {

            AVStream *st = s->streams[pkt->stream_index];

            AVStream *ost = oc->streams[pkt->stream_index];

            st->internal->bsfcs = ost->internal->bsfcs;

            st->internal->nb_bsfcs = ost->internal->nb_bsfcs;

            ost->internal->bsfcs = NULL;

            ost->internal->nb_bsfcs = 0;

        }

        return ret;

    }

    return 1;

}
