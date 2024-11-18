static int mov_read_extradata(MOVContext *c, ByteIOContext *pb, MOVAtom atom)

{

    AVStream *st = c->fc->streams[c->fc->nb_streams-1];

    uint64_t size= (uint64_t)st->codec->extradata_size + atom.size + 8 + FF_INPUT_BUFFER_PADDING_SIZE;

    uint8_t *buf;

    if(size > INT_MAX || (uint64_t)atom.size > INT_MAX)

        return -1;

    buf= av_realloc(st->codec->extradata, size);

    if(!buf)

        return -1;

    st->codec->extradata= buf;

    buf+= st->codec->extradata_size;

    st->codec->extradata_size= size - FF_INPUT_BUFFER_PADDING_SIZE;

    AV_WB32(       buf    , atom.size + 8);

    AV_WL32(       buf + 4, atom.type);

    get_buffer(pb, buf + 8, atom.size);

    return 0;

}
