static int asf_read_value(AVFormatContext *s, uint8_t *name, uint16_t name_len,

                          uint16_t val_len, int type, AVDictionary **met)

{

    int ret;

    uint8_t *value;

    uint16_t buflen = 2 * val_len + 1;

    AVIOContext *pb = s->pb;



    value = av_malloc(buflen);

    if (!value)

        return AVERROR(ENOMEM);

    if (type == ASF_UNICODE) {

        // get_asf_string reads UTF-16 and converts it to UTF-8 which needs longer buffer

        if ((ret = get_asf_string(pb, val_len, value, buflen)) < 0)

            goto failed;

        if (av_dict_set(met, name, value, 0) < 0)

            av_log(s, AV_LOG_WARNING, "av_dict_set failed.\n");

    } else {

        char buf[256];

        if (val_len > sizeof(buf)) {

            ret = AVERROR_INVALIDDATA;

            goto failed;

        }

        if ((ret = avio_read(pb, value, val_len)) < 0)

            goto failed;

        if (ret < 2 * val_len)

            value[ret] = '\0';

        else

            value[2 * val_len - 1] = '\0';

        snprintf(buf, sizeof(buf), "%s", value);

        if (av_dict_set(met, name, buf, 0) < 0)

            av_log(s, AV_LOG_WARNING, "av_dict_set failed.\n");

    }

    av_freep(&value);



    return 0;



failed:

    av_freep(&value);

    return ret;

}
