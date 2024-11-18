int avformat_write_header(AVFormatContext *s, AVDictionary **options)

{

    int ret = 0, i;

    AVStream *st;

    AVDictionary *tmp = NULL;



    if (options)

        av_dict_copy(&tmp, *options, 0);

    if ((ret = av_opt_set_dict(s, &tmp)) < 0)

        goto fail;

    if (s->priv_data && s->oformat->priv_class && *(const AVClass**)s->priv_data==s->oformat->priv_class &&

        (ret = av_opt_set_dict(s->priv_data, &tmp)) < 0)

        goto fail;



    // some sanity checks

    if (s->nb_streams == 0 && !(s->oformat->flags & AVFMT_NOSTREAMS)) {

        av_log(s, AV_LOG_ERROR, "no streams\n");

        ret = AVERROR(EINVAL);

        goto fail;

    }



    for(i=0;i<s->nb_streams;i++) {

        st = s->streams[i];



        switch (st->codec->codec_type) {

        case AVMEDIA_TYPE_AUDIO:

            if(st->codec->sample_rate<=0){

                av_log(s, AV_LOG_ERROR, "sample rate not set\n");

                ret = AVERROR(EINVAL);

                goto fail;

            }

            if(!st->codec->block_align)

                st->codec->block_align = st->codec->channels *

                    av_get_bits_per_sample(st->codec->codec_id) >> 3;

            break;

        case AVMEDIA_TYPE_VIDEO:

            if(st->codec->time_base.num<=0 || st->codec->time_base.den<=0){ //FIXME audio too?

                av_log(s, AV_LOG_ERROR, "time base not set\n");

                ret = AVERROR(EINVAL);

                goto fail;

            }

            if((st->codec->width<=0 || st->codec->height<=0) && !(s->oformat->flags & AVFMT_NODIMENSIONS)){

                av_log(s, AV_LOG_ERROR, "dimensions not set\n");

                ret = AVERROR(EINVAL);

                goto fail;

            }

            if(av_cmp_q(st->sample_aspect_ratio, st->codec->sample_aspect_ratio)

               && FFABS(av_q2d(st->sample_aspect_ratio) - av_q2d(st->codec->sample_aspect_ratio)) > 0.004*av_q2d(st->sample_aspect_ratio)

            ){

                av_log(s, AV_LOG_ERROR, "Aspect ratio mismatch between muxer "

                       "(%d/%d) and encoder layer (%d/%d)\n",

                       st->sample_aspect_ratio.num, st->sample_aspect_ratio.den,

                       st->codec->sample_aspect_ratio.num,

                       st->codec->sample_aspect_ratio.den);

                ret = AVERROR(EINVAL);

                goto fail;

            }

            break;

        }



        if(s->oformat->codec_tag){

            if(st->codec->codec_tag && st->codec->codec_id == CODEC_ID_RAWVIDEO && av_codec_get_tag(s->oformat->codec_tag, st->codec->codec_id) == 0 && !validate_codec_tag(s, st)){

                //the current rawvideo encoding system ends up setting the wrong codec_tag for avi, we override it here

                st->codec->codec_tag= 0;

            }

            if(st->codec->codec_tag){

                if (!validate_codec_tag(s, st)) {

                    char tagbuf[32], cortag[32];

                    av_get_codec_tag_string(tagbuf, sizeof(tagbuf), st->codec->codec_tag);

                    av_get_codec_tag_string(cortag, sizeof(cortag), av_codec_get_tag(s->oformat->codec_tag, st->codec->codec_id));

                    av_log(s, AV_LOG_ERROR,

                           "Tag %s/0x%08x incompatible with output codec id '%d' (%s)\n",

                           tagbuf, st->codec->codec_tag, st->codec->codec_id, cortag);

                    ret = AVERROR_INVALIDDATA;

                    goto fail;

                }

            }else

                st->codec->codec_tag= av_codec_get_tag(s->oformat->codec_tag, st->codec->codec_id);

        }



        if(s->oformat->flags & AVFMT_GLOBALHEADER &&

            !(st->codec->flags & CODEC_FLAG_GLOBAL_HEADER))

          av_log(s, AV_LOG_WARNING, "Codec for stream %d does not use global headers but container format requires global headers\n", i);

    }



    if (!s->priv_data && s->oformat->priv_data_size > 0) {

        s->priv_data = av_mallocz(s->oformat->priv_data_size);

        if (!s->priv_data) {

            ret = AVERROR(ENOMEM);

            goto fail;

        }

        if (s->oformat->priv_class) {

            *(const AVClass**)s->priv_data= s->oformat->priv_class;

            av_opt_set_defaults(s->priv_data);

            if ((ret = av_opt_set_dict(s->priv_data, &tmp)) < 0)

                goto fail;

        }

    }



    /* set muxer identification string */

    if (s->nb_streams && !(s->streams[0]->codec->flags & CODEC_FLAG_BITEXACT)) {

        av_dict_set(&s->metadata, "encoder", LIBAVFORMAT_IDENT, 0);

    }



    if(s->oformat->write_header){

        ret = s->oformat->write_header(s);

        if (ret < 0)

            goto fail;

    }



    /* init PTS generation */

    for(i=0;i<s->nb_streams;i++) {

        int64_t den = AV_NOPTS_VALUE;

        st = s->streams[i];



        switch (st->codec->codec_type) {

        case AVMEDIA_TYPE_AUDIO:

            den = (int64_t)st->time_base.num * st->codec->sample_rate;

            break;

        case AVMEDIA_TYPE_VIDEO:

            den = (int64_t)st->time_base.num * st->codec->time_base.den;

            break;

        default:

            break;

        }

        if (den != AV_NOPTS_VALUE) {

            if (den <= 0) {

                ret = AVERROR_INVALIDDATA;

                goto fail;

            }

            frac_init(&st->pts, 0, 0, den);

        }

    }



    if (options) {

        av_dict_free(options);

        *options = tmp;

    }

    return 0;

fail:

    av_dict_free(&tmp);

    return ret;

}
