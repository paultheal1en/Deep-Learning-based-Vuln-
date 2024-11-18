static int mov_read_glbl(MOVContext *c, ByteIOContext *pb, MOVAtom atom)

{

    AVStream *st = c->fc->streams[c->fc->nb_streams-1];



    if((uint64_t)atom.size > (1<<30))

        return -1;



    av_free(st->codec->extradata);

    st->codec->extradata = av_mallocz(atom.size + FF_INPUT_BUFFER_PADDING_SIZE);

    if (!st->codec->extradata)

        return AVERROR(ENOMEM);

    st->codec->extradata_size = atom.size;

    get_buffer(pb, st->codec->extradata, atom.size);

    return 0;

}
