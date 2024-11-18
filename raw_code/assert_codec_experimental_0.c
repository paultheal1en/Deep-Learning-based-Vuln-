static void assert_codec_experimental(AVCodecContext *c, int encoder)

{

    const char *codec_string = encoder ? "encoder" : "decoder";

    AVCodec *codec;

    if (c->codec->capabilities & CODEC_CAP_EXPERIMENTAL &&

        c->strict_std_compliance > FF_COMPLIANCE_EXPERIMENTAL) {

        av_log(NULL, AV_LOG_FATAL, "%s '%s' is experimental and might produce bad "

                "results.\nAdd '-strict experimental' if you want to use it.\n",

                codec_string, c->codec->name);

        codec = encoder ? avcodec_find_encoder(c->codec->id) : avcodec_find_decoder(c->codec->id);

        if (!(codec->capabilities & CODEC_CAP_EXPERIMENTAL))

            av_log(NULL, AV_LOG_FATAL, "Or use the non experimental %s '%s'.\n",

                   codec_string, codec->name);

        exit(1);

    }

}
