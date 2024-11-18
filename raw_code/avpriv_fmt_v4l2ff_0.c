enum AVPixelFormat avpriv_fmt_v4l2ff(uint32_t v4l2_fmt, enum AVCodecID codec_id)

{

    int i;



    for (i = 0; avpriv_fmt_conversion_table[i].codec_id != AV_CODEC_ID_NONE; i++) {

        if (avpriv_fmt_conversion_table[i].v4l2_fmt == v4l2_fmt &&

            avpriv_fmt_conversion_table[i].codec_id == codec_id) {

            return avpriv_fmt_conversion_table[i].ff_fmt;

        }

    }



    return AV_PIX_FMT_NONE;

}
