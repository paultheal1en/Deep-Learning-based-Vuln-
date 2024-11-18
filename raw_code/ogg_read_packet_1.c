static int ogg_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    struct ogg *ogg;

    struct ogg_stream *os;

    int idx, ret;

    int pstart, psize;

    int64_t fpos, pts, dts;



    if (s->io_repositioned) {

        ogg_reset(s);

        s->io_repositioned = 0;




    //Get an ogg packet

retry:

    do {

        ret = ogg_packet(s, &idx, &pstart, &psize, &fpos);

        if (ret < 0)

            return ret;

    } while (idx < 0 || !s->streams[idx]);



    ogg = s->priv_data;

    os  = ogg->streams + idx;



    // pflags might not be set until after this

    pts = ogg_calc_pts(s, idx, &dts);

    ogg_validate_keyframe(s, idx, pstart, psize);



    if (os->keyframe_seek && !(os->pflags & AV_PKT_FLAG_KEY))

        goto retry;

    os->keyframe_seek = 0;



    //Alloc a pkt

    ret = av_new_packet(pkt, psize);

    if (ret < 0)

        return ret;

    pkt->stream_index = idx;

    memcpy(pkt->data, os->buf + pstart, psize);



    pkt->pts      = pts;

    pkt->dts      = dts;

    pkt->flags    = os->pflags;

    pkt->duration = os->pduration;

    pkt->pos      = fpos;



    if (os->end_trimming) {

        uint8_t *side_data = av_packet_new_side_data(pkt,

                                                     AV_PKT_DATA_SKIP_SAMPLES,

                                                     10);






        AV_WL32(side_data + 4, os->end_trimming);

        os->end_trimming = 0;




    if (os->new_metadata) {

        uint8_t *side_data = av_packet_new_side_data(pkt,

                                                     AV_PKT_DATA_METADATA_UPDATE,

                                                     os->new_metadata_size);






        memcpy(side_data, os->new_metadata, os->new_metadata_size);

        av_freep(&os->new_metadata);

        os->new_metadata_size = 0;




    return psize;
