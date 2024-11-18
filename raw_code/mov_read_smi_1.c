static int mov_read_smi(MOVContext *c, AVIOContext *pb, MOVAtom atom)

{

    AVStream *st;



    if (c->fc->nb_streams < 1)

        return 0;

    st = c->fc->streams[c->fc->nb_streams-1];



    if ((uint64_t)atom.size > (1<<30))

        return AVERROR_INVALIDDATA;



    // currently SVQ3 decoder expect full STSD header - so let's fake it

    // this should be fixed and just SMI header should be passed

    av_free(st->codec->extradata);

    st->codec->extradata = av_mallocz(atom.size + 0x5a + FF_INPUT_BUFFER_PADDING_SIZE);

    if (!st->codec->extradata)

        return AVERROR(ENOMEM);

    st->codec->extradata_size = 0x5a + atom.size;

    memcpy(st->codec->extradata, "SVQ3", 4); // fake

    avio_read(pb, st->codec->extradata + 0x5a, atom.size);

    av_log(c->fc, AV_LOG_TRACE, "Reading SMI %"PRId64"  %s\n", atom.size, st->codec->extradata + 0x5a);

    return 0;

}
