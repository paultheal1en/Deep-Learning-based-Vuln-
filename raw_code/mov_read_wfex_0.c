static int mov_read_wfex(MOVContext *c, AVIOContext *pb, MOVAtom atom)

{

    AVStream *st;



    if (c->fc->nb_streams < 1)

        return 0;

    st = c->fc->streams[c->fc->nb_streams-1];



    ff_get_wav_header(pb, st->codec, atom.size);



    return 0;

}
