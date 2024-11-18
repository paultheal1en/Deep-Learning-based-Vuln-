static int mov_read_wave(MOVContext *c, AVIOContext *pb, MOVAtom atom)

{

    AVStream *st;



    if (c->fc->nb_streams < 1)

        return 0;

    st = c->fc->streams[c->fc->nb_streams-1];



    if ((uint64_t)atom.size > (1<<30))

        return AVERROR_INVALIDDATA;



    if (st->codec->codec_id == AV_CODEC_ID_QDM2 || st->codec->codec_id == AV_CODEC_ID_QDMC) {

        // pass all frma atom to codec, needed at least for QDMC and QDM2

        av_free(st->codec->extradata);

        st->codec->extradata = av_mallocz(atom.size + FF_INPUT_BUFFER_PADDING_SIZE);

        if (!st->codec->extradata)

            return AVERROR(ENOMEM);

        st->codec->extradata_size = atom.size;

        avio_read(pb, st->codec->extradata, atom.size);

    } else if (atom.size > 8) { /* to read frma, esds atoms */

        int ret;

        if ((ret = mov_read_default(c, pb, atom)) < 0)

            return ret;

    } else

        avio_skip(pb, atom.size);

    return 0;

}
