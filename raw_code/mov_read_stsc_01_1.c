static int mov_read_stsc(MOVContext *c, ByteIOContext *pb, MOV_atom_t atom)

{

    AVStream *st = c->fc->streams[c->fc->nb_streams-1];

    MOVStreamContext *sc = (MOVStreamContext *)st->priv_data;

    int entries, i;



    print_atom("stsc", atom);



    get_byte(pb); /* version */

    get_byte(pb); get_byte(pb); get_byte(pb); /* flags */



    entries = get_be32(pb);

#ifdef DEBUG

av_log(NULL, AV_LOG_DEBUG, "track[%i].stsc.entries = %i\n", c->fc->nb_streams-1, entries);

#endif

    sc->sample_to_chunk_sz = entries;

    sc->sample_to_chunk = (MOV_sample_to_chunk_tbl*) av_malloc(entries * sizeof(MOV_sample_to_chunk_tbl));

    if (!sc->sample_to_chunk)

        return -1;

    for(i=0; i<entries; i++) {

        sc->sample_to_chunk[i].first = get_be32(pb);

        sc->sample_to_chunk[i].count = get_be32(pb);

        sc->sample_to_chunk[i].id = get_be32(pb);

#ifdef DEBUG

/*        av_log(NULL, AV_LOG_DEBUG, "sample_to_chunk first=%ld count=%ld, id=%ld\n", sc->sample_to_chunk[i].first, sc->sample_to_chunk[i].count, sc->sample_to_chunk[i].id); */

#endif

    }

    return 0;

}
