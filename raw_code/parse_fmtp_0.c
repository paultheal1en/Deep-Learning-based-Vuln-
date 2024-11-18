static int parse_fmtp(AVFormatContext *s,

                      AVStream *stream, PayloadContext *data,

                      const char *attr, const char *value)

{

    AVCodecParameters *par = stream->codecpar;

    int res, i;



    if (!strcmp(attr, "config")) {

        res = parse_fmtp_config(par, value);



        if (res < 0)

            return res;

    }



    if (par->codec_id == AV_CODEC_ID_AAC) {

        /* Looking for a known attribute */

        for (i = 0; attr_names[i].str; ++i) {

            if (!av_strcasecmp(attr, attr_names[i].str)) {

                if (attr_names[i].type == ATTR_NAME_TYPE_INT) {

                    *(int *)((char *)data+

                        attr_names[i].offset) = atoi(value);

                } else if (attr_names[i].type == ATTR_NAME_TYPE_STR)

                    *(char **)((char *)data+

                        attr_names[i].offset) = av_strdup(value);

            }

        }

    }

    return 0;

}
