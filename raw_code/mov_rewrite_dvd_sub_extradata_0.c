static int mov_rewrite_dvd_sub_extradata(AVStream *st)

{

    char pal_s[256];

    char buf[256];

    int pal_s_pos = 0;

    uint8_t *src = st->codec->extradata;

    int i;



    if (st->codec->extradata_size != 64)

        return 0;



    for (i = 0; i < 16; i++) {

        uint32_t yuv = AV_RB32(src + i * 4);

        uint32_t rgba = yuv_to_rgba(yuv);



        snprintf(pal_s + pal_s_pos, sizeof(pal_s) - pal_s_pos, "%06x%s", rgba,

                 i != 15 ? ", " : "");

        pal_s_pos = strlen(pal_s);

        if (pal_s_pos >= sizeof(pal_s))

            return 0;

    }



    snprintf(buf, sizeof(buf), "size: %dx%d\npalette: %s\n",

             st->codec->width, st->codec->height, pal_s);



    av_freep(&st->codec->extradata);

    st->codec->extradata_size = 0;

    st->codec->extradata = av_mallocz(strlen(buf) + FF_INPUT_BUFFER_PADDING_SIZE);

    if (!st->codec->extradata)

        return AVERROR(ENOMEM);

    st->codec->extradata_size = strlen(buf);

    memcpy(st->codec->extradata, buf, st->codec->extradata_size);



    return 0;

}
