flac_header (AVFormatContext * s, int idx)

{

    struct ogg *ogg = s->priv_data;

    struct ogg_stream *os = ogg->streams + idx;

    AVStream *st = s->streams[idx];

    GetBitContext gb;

    FLACStreaminfo si;

    int mdt;



    if (os->buf[os->pstart] == 0xff)

        return 0;



    init_get_bits(&gb, os->buf + os->pstart, os->psize*8);

    skip_bits1(&gb); /* metadata_last */

    mdt = get_bits(&gb, 7);



    if (mdt == OGG_FLAC_METADATA_TYPE_STREAMINFO) {

        uint8_t *streaminfo_start = os->buf + os->pstart + 5 + 4 + 4 + 4;

        skip_bits_long(&gb, 4*8); /* "FLAC" */

        if(get_bits(&gb, 8) != 1) /* unsupported major version */

            return -1;

        skip_bits_long(&gb, 8 + 16); /* minor version + header count */

        skip_bits_long(&gb, 4*8); /* "fLaC" */



        /* METADATA_BLOCK_HEADER */

        if (get_bits_long(&gb, 32) != FLAC_STREAMINFO_SIZE)

            return -1;



        avpriv_flac_parse_streaminfo(st->codec, &si, streaminfo_start);



        st->codec->codec_type = AVMEDIA_TYPE_AUDIO;

        st->codec->codec_id = AV_CODEC_ID_FLAC;

        st->need_parsing = AVSTREAM_PARSE_HEADERS;



        ff_alloc_extradata(st->codec, FLAC_STREAMINFO_SIZE);

        memcpy(st->codec->extradata, streaminfo_start, st->codec->extradata_size);



        avpriv_set_pts_info(st, 64, 1, st->codec->sample_rate);

    } else if (mdt == FLAC_METADATA_TYPE_VORBIS_COMMENT) {

        ff_vorbis_comment (s, &st->metadata, os->buf + os->pstart + 4, os->psize - 4);

    }



    return 1;

}
