static int mov_read_stsd(MOVContext *c, AVIOContext *pb, MOVAtom atom)

{

    AVStream *st;

    MOVStreamContext *sc;

    int ret;



    if (c->fc->nb_streams < 1)

        return 0;

    st = c->fc->streams[c->fc->nb_streams - 1];

    sc = st->priv_data;



    avio_r8(pb); /* version */

    avio_rb24(pb); /* flags */

    sc->stsd_count = avio_rb32(pb); /* entries */



    /* Prepare space for hosting multiple extradata. */

    sc->extradata = av_mallocz_array(sc->stsd_count, sizeof(*sc->extradata));

    if (!sc->extradata)

        return AVERROR(ENOMEM);



    sc->extradata_size = av_mallocz_array(sc->stsd_count, sizeof(*sc->extradata_size));

    if (!sc->extradata_size)

        return AVERROR(ENOMEM);



    ret = ff_mov_read_stsd_entries(c, pb, sc->stsd_count);

    if (ret < 0)

        return ret;



    /* Restore back the primary extradata. */

    av_freep(&st->codecpar->extradata);

    st->codecpar->extradata_size = sc->extradata_size[0];

    if (sc->extradata_size[0]) {

        st->codecpar->extradata = av_mallocz(sc->extradata_size[0] + AV_INPUT_BUFFER_PADDING_SIZE);

        if (!st->codecpar->extradata)

            return AVERROR(ENOMEM);

        memcpy(st->codecpar->extradata, sc->extradata[0], sc->extradata_size[0]);

    }



    return 0;

}
