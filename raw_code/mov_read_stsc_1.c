static int mov_read_stsc(MOVContext *c, AVIOContext *pb, MOVAtom atom)

{

    AVStream *st;

    MOVStreamContext *sc;

    unsigned int i, entries;



    if (c->fc->nb_streams < 1)

        return 0;

    st = c->fc->streams[c->fc->nb_streams-1];

    sc = st->priv_data;



    avio_r8(pb); /* version */

    avio_rb24(pb); /* flags */



    entries = avio_rb32(pb);



    av_log(c->fc, AV_LOG_TRACE, "track[%i].stsc.entries = %i\n", c->fc->nb_streams-1, entries);



    if (!entries)

        return 0;

    if (entries >= UINT_MAX / sizeof(*sc->stsc_data))

        return AVERROR_INVALIDDATA;

    sc->stsc_data = av_malloc(entries * sizeof(*sc->stsc_data));

    if (!sc->stsc_data)

        return AVERROR(ENOMEM);



    for (i = 0; i < entries && !pb->eof_reached; i++) {

        sc->stsc_data[i].first = avio_rb32(pb);

        sc->stsc_data[i].count = avio_rb32(pb);

        sc->stsc_data[i].id = avio_rb32(pb);

        if (sc->stsc_data[i].id > sc->stsd_count)

            return AVERROR_INVALIDDATA;

    }



    sc->stsc_count = i;



    if (pb->eof_reached)

        return AVERROR_EOF;



    return 0;

}
