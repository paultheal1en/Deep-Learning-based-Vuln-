static int mov_read_stco(MOVContext *c, ByteIOContext *pb, MOVAtom atom)

{

    AVStream *st = c->fc->streams[c->fc->nb_streams-1];

    MOVStreamContext *sc = st->priv_data;

    unsigned int i, entries;



    get_byte(pb); /* version */

    get_be24(pb); /* flags */



    entries = get_be32(pb);



    if(entries >= UINT_MAX/sizeof(int64_t))

        return -1;



    sc->chunk_offsets = av_malloc(entries * sizeof(int64_t));

    if (!sc->chunk_offsets)

        return AVERROR(ENOMEM);

    sc->chunk_count = entries;



    if      (atom.type == MKTAG('s','t','c','o'))

        for(i=0; i<entries; i++)

            sc->chunk_offsets[i] = get_be32(pb);

    else if (atom.type == MKTAG('c','o','6','4'))

        for(i=0; i<entries; i++)

            sc->chunk_offsets[i] = get_be64(pb);

    else

        return -1;



    return 0;

}
