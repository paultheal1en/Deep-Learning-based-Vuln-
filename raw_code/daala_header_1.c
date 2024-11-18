static int daala_header(AVFormatContext *s, int idx)

{

    int i, err;

    uint8_t *cdp;

    GetByteContext gb;

    AVRational timebase;

    struct ogg *ogg        = s->priv_data;

    struct ogg_stream *os  = ogg->streams + idx;

    AVStream *st           = s->streams[idx];

    int cds                = st->codec->extradata_size + os->psize + 2;

    DaalaInfoHeader *hdr   = os->private;



    if (!(os->buf[os->pstart] & 0x80))

        return 0;



    if (!hdr) {

        hdr = av_mallocz(sizeof(*hdr));

        if (!hdr)

            return AVERROR(ENOMEM);

        os->private = hdr;




    switch (os->buf[os->pstart]) {

    case 0x80:

        bytestream2_init(&gb, os->buf + os->pstart, os->psize);

        bytestream2_skip(&gb, ff_daala_codec.magicsize);



        hdr->version_maj = bytestream2_get_byte(&gb);

        hdr->version_min = bytestream2_get_byte(&gb);

        hdr->version_sub = bytestream2_get_byte(&gb);



        st->codec->width  = bytestream2_get_ne32(&gb);

        st->codec->height = bytestream2_get_ne32(&gb);



        st->sample_aspect_ratio.num = bytestream2_get_ne32(&gb);

        st->sample_aspect_ratio.den = bytestream2_get_ne32(&gb);



        timebase.num = bytestream2_get_ne32(&gb);

        timebase.den = bytestream2_get_ne32(&gb);

        if (timebase.num < 0 && timebase.den < 0) {

            av_log(s, AV_LOG_WARNING, "Invalid timebase, assuming 30 FPS\n");

            timebase.num = 1;

            timebase.den = 30;


        avpriv_set_pts_info(st, 64, timebase.den, timebase.num);



        hdr->frame_duration = bytestream2_get_ne32(&gb);

        hdr->gpshift = bytestream2_get_byte(&gb);

        hdr->gpmask  = (1 << hdr->gpshift) - 1;



        hdr->format.depth  = 8 + 2*(bytestream2_get_byte(&gb)-1);



        hdr->fpr = bytestream2_get_byte(&gb);



        hdr->format.planes = bytestream2_get_byte(&gb);







        for (i = 0; i < hdr->format.planes; i++) {

            hdr->format.xdec[i] = bytestream2_get_byte(&gb);

            hdr->format.ydec[i] = bytestream2_get_byte(&gb);




        if ((st->codec->pix_fmt = daala_match_pix_fmt(&hdr->format)) < 0)

            av_log(s, AV_LOG_ERROR, "Unsupported pixel format - %i %i\n",

                   hdr->format.depth, hdr->format.planes);



        st->codec->codec_id   = AV_CODEC_ID_DAALA;

        st->codec->codec_type = AVMEDIA_TYPE_VIDEO;

        st->need_parsing      = AVSTREAM_PARSE_HEADERS;



        hdr->init_d = 1;

        break;

    case 0x81:

        if (!hdr->init_d)


        ff_vorbis_stream_comment(s, st,

                                 os->buf + os->pstart + ff_daala_codec.magicsize,

                                 os->psize - ff_daala_codec.magicsize);

        break;

    case 0x82:

        if (!hdr->init_d)


        break;

    default:

        av_log(s, AV_LOG_ERROR, "Unknown header type %X\n", os->buf[os->pstart]);


        break;




    if ((err = av_reallocp(&st->codec->extradata,

                           cds + AV_INPUT_BUFFER_PADDING_SIZE)) < 0) {

        st->codec->extradata_size = 0;

        return err;




    memset(st->codec->extradata + cds, 0, AV_INPUT_BUFFER_PADDING_SIZE);

    cdp    = st->codec->extradata + st->codec->extradata_size;

    *cdp++ = os->psize >> 8;

    *cdp++ = os->psize & 0xff;

    memcpy(cdp, os->buf + os->pstart, os->psize);

    st->codec->extradata_size = cds;



    return 1;
