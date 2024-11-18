static int mov_read_wave(MOVContext *c, ByteIOContext *pb, MOV_atom_t atom)

{

    AVStream *st = c->fc->streams[c->fc->nb_streams-1];



    if((uint64_t)atom.size > (1<<30))

        return -1;



    if (st->codec->codec_id == CODEC_ID_QDM2) {

        // pass all frma atom to codec, needed at least for QDM2

        av_free(st->codec->extradata);

        st->codec->extradata_size = atom.size;

        st->codec->extradata = av_mallocz(st->codec->extradata_size + FF_INPUT_BUFFER_PADDING_SIZE);



        if (st->codec->extradata) {

            get_buffer(pb, st->codec->extradata, atom.size);

        } else

            url_fskip(pb, atom.size);

    } else if (atom.size > 8) { /* to read frma, esds atoms */

        mov_read_default(c, pb, atom);

    } else

        url_fskip(pb, atom.size);

    return 0;

}
