static void show_stream(AVFormatContext *fmt_ctx, int stream_idx)

{

    AVStream *stream = fmt_ctx->streams[stream_idx];

    AVCodecContext *dec_ctx;

    AVCodec *dec;

    char val_str[128];

    AVDictionaryEntry *tag = NULL;

    AVRational display_aspect_ratio;



    printf("[STREAM]\n");



    printf("index=%d\n", stream->index);



    if ((dec_ctx = stream->codec)) {

        if ((dec = dec_ctx->codec)) {

            printf("codec_name=%s\n", dec->name);

            printf("codec_long_name=%s\n", dec->long_name);

        } else {

            printf("codec_name=unknown\n");

        }



        printf("codec_type=%s\n", media_type_string(dec_ctx->codec_type));

        printf("codec_time_base=%d/%d\n",

               dec_ctx->time_base.num, dec_ctx->time_base.den);



        /* print AVI/FourCC tag */

        av_get_codec_tag_string(val_str, sizeof(val_str), dec_ctx->codec_tag);

        printf("codec_tag_string=%s\n", val_str);

        printf("codec_tag=0x%04x\n", dec_ctx->codec_tag);



        switch (dec_ctx->codec_type) {

        case AVMEDIA_TYPE_VIDEO:

            printf("width=%d\n", dec_ctx->width);

            printf("height=%d\n", dec_ctx->height);

            printf("has_b_frames=%d\n", dec_ctx->has_b_frames);

            if (dec_ctx->sample_aspect_ratio.num) {

                printf("sample_aspect_ratio=%d:%d\n",

                       dec_ctx->sample_aspect_ratio.num,

                       dec_ctx->sample_aspect_ratio.den);

                av_reduce(&display_aspect_ratio.num, &display_aspect_ratio.den,

                          dec_ctx->width  * dec_ctx->sample_aspect_ratio.num,

                          dec_ctx->height * dec_ctx->sample_aspect_ratio.den,

                          1024*1024);

                printf("display_aspect_ratio=%d:%d\n",

                       display_aspect_ratio.num, display_aspect_ratio.den);

            }

            printf("pix_fmt=%s\n",

                   dec_ctx->pix_fmt != PIX_FMT_NONE ? av_pix_fmt_descriptors[dec_ctx->pix_fmt].name

                                                    : "unknown");

            printf("level=%d\n", dec_ctx->level);

            break;



        case AVMEDIA_TYPE_AUDIO:

            printf("sample_rate=%s\n", value_string(val_str, sizeof(val_str),

                                                    dec_ctx->sample_rate,

                                                    unit_hertz_str));

            printf("channels=%d\n", dec_ctx->channels);

            printf("bits_per_sample=%d\n",

                   av_get_bits_per_sample(dec_ctx->codec_id));

            break;

        }

    } else {

        printf("codec_type=unknown\n");

    }



    if (fmt_ctx->iformat->flags & AVFMT_SHOW_IDS)

        printf("id=0x%x\n", stream->id);

    printf("r_frame_rate=%d/%d\n",

           stream->r_frame_rate.num, stream->r_frame_rate.den);

    printf("avg_frame_rate=%d/%d\n",

           stream->avg_frame_rate.num, stream->avg_frame_rate.den);

    printf("time_base=%d/%d\n",

           stream->time_base.num, stream->time_base.den);

    printf("start_time=%s\n",

           time_value_string(val_str, sizeof(val_str),

                             stream->start_time, &stream->time_base));

    printf("duration=%s\n",

           time_value_string(val_str, sizeof(val_str),

                             stream->duration, &stream->time_base));

    if (stream->nb_frames)

        printf("nb_frames=%"PRId64"\n", stream->nb_frames);



    while ((tag = av_dict_get(stream->metadata, "", tag,

                              AV_DICT_IGNORE_SUFFIX)))

        printf("TAG:%s=%s\n", tag->key, tag->value);



    printf("[/STREAM]\n");

}
