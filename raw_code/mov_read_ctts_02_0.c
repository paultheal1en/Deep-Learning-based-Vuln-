static int mov_read_ctts(MOVContext *c, ByteIOContext *pb, MOVAtom atom)

{

    AVStream *st = c->fc->streams[c->fc->nb_streams-1];

    MOVStreamContext *sc = st->priv_data;

    unsigned int i, entries;



    get_byte(pb); /* version */

    get_be24(pb); /* flags */

    entries = get_be32(pb);



    dprintf(c->fc, "track[%i].ctts.entries = %i\n", c->fc->nb_streams-1, entries);



    if(entries >= UINT_MAX / sizeof(*sc->ctts_data))

        return -1;

    sc->ctts_data = av_malloc(entries * sizeof(*sc->ctts_data));

    if (!sc->ctts_data)

        return AVERROR(ENOMEM);

    sc->ctts_count = entries;



    for(i=0; i<entries; i++) {

        int count    =get_be32(pb);

        int duration =get_be32(pb);



        sc->ctts_data[i].count   = count;

        sc->ctts_data[i].duration= duration;

    }

    return 0;

}
