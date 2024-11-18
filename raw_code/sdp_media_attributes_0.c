static char *sdp_media_attributes(char *buff, int size, AVCodecContext *c, int payload_type)

{

    char *config = NULL;



    switch (c->codec_id) {

        case CODEC_ID_MPEG4:

            if (c->flags & CODEC_FLAG_GLOBAL_HEADER) {

                config = extradata2config(c->extradata, c->extradata_size);

            }

            av_strlcatf(buff, size, "a=rtpmap:%d MP4V-ES/90000\r\n"

                                    "a=fmtp:%d profile-level-id=1%s\r\n",

                                     payload_type,

                                     payload_type, config ? config : "");

            break;

        case CODEC_ID_AAC:

            if (c->flags & CODEC_FLAG_GLOBAL_HEADER) {

                config = extradata2config(c->extradata, c->extradata_size);

            } else {

                /* FIXME: maybe we can forge config information based on the

                 *        codec parameters...

                 */

                av_log(NULL, AV_LOG_ERROR, "AAC with no global headers is currently not supported\n");

                return NULL;

            }

            if (config == NULL) {

                return NULL;

            }

            av_strlcatf(buff, size, "a=rtpmap:%d MPEG4-GENERIC/%d/%d\r\n"

                                    "a=fmtp:%d profile-level-id=1;"

                                    "mode=AAC-hbr;sizelength=13;indexlength=3;"

                                    "indexdeltalength=3%s\r\n",

                                     payload_type, c->sample_rate, c->channels,

                                     payload_type, config);

            break;

        default:

            /* Nothing special to do, here... */

            break;

    }



    av_free(config);



    return buff;

}
