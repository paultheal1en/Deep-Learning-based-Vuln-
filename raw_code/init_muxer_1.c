static int init_muxer(AVFormatContext *s, AVDictionary **options)

{

    int ret = 0, i;

    AVStream *st;

    AVDictionary *tmp = NULL;

    AVCodecParameters *par = NULL;

    AVOutputFormat *of = s->oformat;

    const AVCodecDescriptor *desc;

    AVDictionaryEntry *e;



    if (options)

        av_dict_copy(&tmp, *options, 0);



    if ((ret = av_opt_set_dict(s, &tmp)) < 0)

        goto fail;

    if (s->priv_data && s->oformat->priv_class && *(const AVClass**)s->priv_data==s->oformat->priv_class &&

        (ret = av_opt_set_dict2(s->priv_data, &tmp, AV_OPT_SEARCH_CHILDREN)) < 0)

        goto fail;



#if FF_API_LAVF_AVCTX

FF_DISABLE_DEPRECATION_WARNINGS

    if (s->nb_streams && s->streams[0]->codec->flags & AV_CODEC_FLAG_BITEXACT) {

        if (!(s->flags & AVFMT_FLAG_BITEXACT)) {

#if FF_API_LAVF_BITEXACT

            av_log(s, AV_LOG_WARNING,

                   "Setting the AVFormatContext to bitexact mode, because "

                   "the AVCodecContext is in that mode. This behavior will "

                   "change in the future. To keep the current behavior, set "

                   "AVFormatContext.flags |= AVFMT_FLAG_BITEXACT.\n");

            s->flags |= AVFMT_FLAG_BITEXACT;

#else

            av_log(s, AV_LOG_WARNING,

                   "The AVFormatContext is not in set to bitexact mode, only "

                   "the AVCodecContext. If this is not intended, set "

                   "AVFormatContext.flags |= AVFMT_FLAG_BITEXACT.\n");

#endif

        }

    }

FF_ENABLE_DEPRECATION_WARNINGS

#endif



    // some sanity checks

    if (s->nb_streams == 0 && !(of->flags & AVFMT_NOSTREAMS)) {

        av_log(s, AV_LOG_ERROR, "No streams to mux were specified\n");

        ret = AVERROR(EINVAL);

        goto fail;

    }



    for (i = 0; i < s->nb_streams; i++) {

        st  = s->streams[i];

        par = st->codecpar;



#if FF_API_LAVF_CODEC_TB

FF_DISABLE_DEPRECATION_WARNINGS

        if (!st->time_base.num && st->codec->time_base.num) {

            av_log(s, AV_LOG_WARNING, "Using AVStream.codec.time_base as a "

                   "timebase hint to the muxer is deprecated. Set "

                   "AVStream.time_base instead.\n");

            avpriv_set_pts_info(st, 64, st->codec->time_base.num, st->codec->time_base.den);

        }

FF_ENABLE_DEPRECATION_WARNINGS

#endif



#if FF_API_LAVF_AVCTX

FF_DISABLE_DEPRECATION_WARNINGS

        if (st->codecpar->codec_type == AVMEDIA_TYPE_UNKNOWN &&

            st->codec->codec_type    != AVMEDIA_TYPE_UNKNOWN) {

            av_log(s, AV_LOG_WARNING, "Using AVStream.codec to pass codec "

                   "parameters to muxers is deprecated, use AVStream.codecpar "

                   "instead.\n");

            ret = avcodec_parameters_from_context(st->codecpar, st->codec);

            if (ret < 0)

                goto fail;

        }

FF_ENABLE_DEPRECATION_WARNINGS

#endif



        /* update internal context from codecpar, old bsf api needs this

         * FIXME: remove when autobsf uses new bsf API */

        ret = avcodec_parameters_to_context(st->internal->avctx, st->codecpar);

        if (ret < 0)

            goto fail;



        if (!st->time_base.num) {

            /* fall back on the default timebase values */

            if (par->codec_type == AVMEDIA_TYPE_AUDIO && par->sample_rate)

                avpriv_set_pts_info(st, 64, 1, par->sample_rate);

            else

                avpriv_set_pts_info(st, 33, 1, 90000);

        }



        switch (par->codec_type) {

        case AVMEDIA_TYPE_AUDIO:

            if (par->sample_rate <= 0) {

                av_log(s, AV_LOG_ERROR, "sample rate not set\n");

                ret = AVERROR(EINVAL);

                goto fail;

            }

            if (!par->block_align)

                par->block_align = par->channels *

                                   av_get_bits_per_sample(par->codec_id) >> 3;

            break;

        case AVMEDIA_TYPE_VIDEO:

            if ((par->width <= 0 || par->height <= 0) &&

                !(of->flags & AVFMT_NODIMENSIONS)) {

                av_log(s, AV_LOG_ERROR, "dimensions not set\n");

                ret = AVERROR(EINVAL);

                goto fail;

            }

            if (av_cmp_q(st->sample_aspect_ratio, par->sample_aspect_ratio)

                && fabs(av_q2d(st->sample_aspect_ratio) - av_q2d(par->sample_aspect_ratio)) > 0.004*av_q2d(st->sample_aspect_ratio)

            ) {

                if (st->sample_aspect_ratio.num != 0 &&

                    st->sample_aspect_ratio.den != 0 &&

                    par->sample_aspect_ratio.num != 0 &&

                    par->sample_aspect_ratio.den != 0) {

                    av_log(s, AV_LOG_ERROR, "Aspect ratio mismatch between muxer "

                           "(%d/%d) and encoder layer (%d/%d)\n",

                           st->sample_aspect_ratio.num, st->sample_aspect_ratio.den,

                           par->sample_aspect_ratio.num,

                           par->sample_aspect_ratio.den);

                    ret = AVERROR(EINVAL);

                    goto fail;

                }

            }

            break;

        }



        desc = avcodec_descriptor_get(par->codec_id);

        if (desc && desc->props & AV_CODEC_PROP_REORDER)

            st->internal->reorder = 1;



        if (of->codec_tag) {

            if (   par->codec_tag

                && par->codec_id == AV_CODEC_ID_RAWVIDEO

                && (   av_codec_get_tag(of->codec_tag, par->codec_id) == 0

                    || av_codec_get_tag(of->codec_tag, par->codec_id) == MKTAG('r', 'a', 'w', ' '))

                && !validate_codec_tag(s, st)) {

                // the current rawvideo encoding system ends up setting

                // the wrong codec_tag for avi/mov, we override it here

                par->codec_tag = 0;

            }

            if (par->codec_tag) {

                if (!validate_codec_tag(s, st)) {

                    char tagbuf[32], tagbuf2[32];

                    av_get_codec_tag_string(tagbuf, sizeof(tagbuf), par->codec_tag);

                    av_get_codec_tag_string(tagbuf2, sizeof(tagbuf2), av_codec_get_tag(s->oformat->codec_tag, par->codec_id));

                    av_log(s, AV_LOG_ERROR,

                           "Tag %s/0x%08x incompatible with output codec id '%d' (%s)\n",

                           tagbuf, par->codec_tag, par->codec_id, tagbuf2);

                    ret = AVERROR_INVALIDDATA;

                    goto fail;

                }

            } else

                par->codec_tag = av_codec_get_tag(of->codec_tag, par->codec_id);

        }



        if (par->codec_type != AVMEDIA_TYPE_ATTACHMENT)

            s->internal->nb_interleaved_streams++;

    }



    if (!s->priv_data && of->priv_data_size > 0) {

        s->priv_data = av_mallocz(of->priv_data_size);

        if (!s->priv_data) {

            ret = AVERROR(ENOMEM);

            goto fail;

        }

        if (of->priv_class) {

            *(const AVClass **)s->priv_data = of->priv_class;

            av_opt_set_defaults(s->priv_data);

            if ((ret = av_opt_set_dict2(s->priv_data, &tmp, AV_OPT_SEARCH_CHILDREN)) < 0)

                goto fail;

        }

    }



    /* set muxer identification string */

    if (!(s->flags & AVFMT_FLAG_BITEXACT)) {

        av_dict_set(&s->metadata, "encoder", LIBAVFORMAT_IDENT, 0);

    } else {

        av_dict_set(&s->metadata, "encoder", NULL, 0);

    }



    for (e = NULL; e = av_dict_get(s->metadata, "encoder-", e, AV_DICT_IGNORE_SUFFIX); ) {

        av_dict_set(&s->metadata, e->key, NULL, 0);

    }



    if (options) {

         av_dict_free(options);

         *options = tmp;

    }



    if (s->oformat->init && (ret = s->oformat->init(s)) < 0) {

        s->oformat->deinit(s);

        goto fail;

    }



    return 0;



fail:

    av_dict_free(&tmp);

    return ret;

}
