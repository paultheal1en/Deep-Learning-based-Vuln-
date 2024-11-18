int attribute_align_arg avcodec_open(AVCodecContext *avctx, AVCodec *codec)

{

    int ret = 0;



    /* If there is a user-supplied mutex locking routine, call it. */

    if (ff_lockmgr_cb) {

        if ((*ff_lockmgr_cb)(&codec_mutex, AV_LOCK_OBTAIN))

            return -1;

    }



    entangled_thread_counter++;

    if(entangled_thread_counter != 1){

        av_log(avctx, AV_LOG_ERROR, "insufficient thread locking around avcodec_open/close()\n");

        ret = -1;

        goto end;

    }



    if(avctx->codec || !codec) {

        ret = AVERROR(EINVAL);

        goto end;

    }



    if (codec->priv_data_size > 0) {

      if(!avctx->priv_data){

        avctx->priv_data = av_mallocz(codec->priv_data_size);

        if (!avctx->priv_data) {

            ret = AVERROR(ENOMEM);

            goto end;

        }

        if(codec->priv_class){ //this can be droped once all user apps use   avcodec_get_context_defaults3()

            *(AVClass**)avctx->priv_data= codec->priv_class;

            av_opt_set_defaults(avctx->priv_data);

        }

      }

    } else {

        avctx->priv_data = NULL;

    }



    if(avctx->coded_width && avctx->coded_height)

        avcodec_set_dimensions(avctx, avctx->coded_width, avctx->coded_height);

    else if(avctx->width && avctx->height)

        avcodec_set_dimensions(avctx, avctx->width, avctx->height);



    if ((avctx->coded_width || avctx->coded_height || avctx->width || avctx->height)

        && (  av_image_check_size(avctx->coded_width, avctx->coded_height, 0, avctx) < 0

           || av_image_check_size(avctx->width,       avctx->height,       0, avctx) < 0)) {

        av_log(avctx, AV_LOG_WARNING, "ignoring invalid width/height values\n");

        avcodec_set_dimensions(avctx, 0, 0);

    }



    /* if the decoder init function was already called previously,

       free the already allocated subtitle_header before overwriting it */

    if (codec->decode)

        av_freep(&avctx->subtitle_header);



#define SANE_NB_CHANNELS 128U

    if (avctx->channels > SANE_NB_CHANNELS) {

        ret = AVERROR(EINVAL);

        goto free_and_end;

    }



    avctx->codec = codec;

    if ((avctx->codec_type == AVMEDIA_TYPE_UNKNOWN || avctx->codec_type == codec->type) &&

        avctx->codec_id == CODEC_ID_NONE) {

        avctx->codec_type = codec->type;

        avctx->codec_id   = codec->id;

    }

    if (avctx->codec_id != codec->id || (avctx->codec_type != codec->type

                           && avctx->codec_type != AVMEDIA_TYPE_ATTACHMENT)) {

        av_log(avctx, AV_LOG_ERROR, "codec type or id mismatches\n");

        ret = AVERROR(EINVAL);

        goto free_and_end;

    }

    avctx->frame_number = 0;



    if (HAVE_THREADS && !avctx->thread_opaque) {

        ret = ff_thread_init(avctx);

        if (ret < 0) {

            goto free_and_end;

        }

    }



    if (avctx->codec->max_lowres < avctx->lowres) {

        av_log(avctx, AV_LOG_ERROR, "The maximum value for lowres supported by the decoder is %d\n",

               avctx->codec->max_lowres);

        ret = AVERROR(EINVAL);

        goto free_and_end;

    }

    if (avctx->codec->sample_fmts && avctx->codec->encode) {

        int i;

        for (i = 0; avctx->codec->sample_fmts[i] != AV_SAMPLE_FMT_NONE; i++)

            if (avctx->sample_fmt == avctx->codec->sample_fmts[i])

                break;

        if (avctx->codec->sample_fmts[i] == AV_SAMPLE_FMT_NONE) {

            av_log(avctx, AV_LOG_ERROR, "Specified sample_fmt is not supported.\n");

            ret = AVERROR(EINVAL);

            goto free_and_end;

        }

    }



    if(avctx->codec->init && !(avctx->active_thread_type&FF_THREAD_FRAME)){

        ret = avctx->codec->init(avctx);

        if (ret < 0) {

            goto free_and_end;

        }

    }

end:

    entangled_thread_counter--;



    /* Release any user-supplied mutex. */

    if (ff_lockmgr_cb) {

        (*ff_lockmgr_cb)(&codec_mutex, AV_LOCK_RELEASE);

    }

    return ret;

free_and_end:

    av_freep(&avctx->priv_data);

    avctx->codec= NULL;

    goto end;

}
