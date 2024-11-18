static int mov_read_chan(MOVContext *c, AVIOContext *pb, MOVAtom atom)

{

    AVStream *st;



    if (c->fc->nb_streams < 1)

        return 0;

    st = c->fc->streams[c->fc->nb_streams-1];



    if (atom.size < 16)

        return 0;



    ff_mov_read_chan(c->fc, st, atom.size - 4);



    return 0;

}
