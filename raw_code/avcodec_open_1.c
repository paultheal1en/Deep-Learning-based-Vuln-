int attribute_align_arg avcodec_open(AVCodecContext *avctx, AVCodec *codec)

{

    int ret= -1;



    /* If there is a user-supplied mutex locking routine, call it. */

    if (ff_lockmgr_cb) {

        if ((*ff_lockmgr_cb)(&codec_mutex, AV_LOCK_OBTAIN))

            return -1;

    }



    entangled_thread_counter++;

    if(entangled_thread_counter != 1){

        av_log(avctx, AV_LOG_ERROR, "insufficient thread locking around avcodec_open/close()\n");

        goto end;

    }



    if(avctx->codec || !codec)

        goto end;



    if (codec->priv_data_size > 0) {

        avctx->priv_data = av_mallocz(codec->priv_data_size);

        if (!avctx->priv_data) {

            ret = AVERROR(ENOMEM);

            goto end;

        }

    } else {

        avctx->priv_data = NULL;

    }



    if(avctx->coded_width && avctx->coded_height)

        avcodec_set_dimensions(avctx, avctx->coded_width, avctx->coded_height);

    else if(avctx->width && avctx->height)

        avcodec_set_dimensions(avctx, avctx->width, avctx->height);



#define SANE_NB_CHANNELS 128U

    if (((avctx->coded_width || avctx->coded_height)

        && av_image_check_size(avctx->coded_width, avctx->coded_height, 0, avctx))

        || avctx->channels > SANE_NB_CHANNELS) {

        ret = AVERROR(EINVAL);

        goto free_and_end;

    }



    avctx->codec = codec;

    if ((avctx->codec_type == AVMEDIA_TYPE_UNKNOWN || avctx->codec_type == codec->type) &&

        avctx->codec_id == CODEC_ID_NONE) {

        avctx->codec_type = codec->type;

        avctx->codec_id   = codec->id;

    }

    if(avctx->codec_id != codec->id || avctx->codec_type != codec->type){

        av_log(avctx, AV_LOG_ERROR, "codec type or id mismatches\n");

        goto free_and_end;

    }

    avctx->frame_number = 0;

    if (avctx->codec->max_lowres < avctx->lowres) {

        av_log(avctx, AV_LOG_ERROR, "The maximum value for lowres supported by the decoder is %d\n",

               avctx->codec->max_lowres);

        goto free_and_end;

    }



    if(avctx->codec->init){

        ret = avctx->codec->init(avctx);

        if (ret < 0) {

            goto free_and_end;

        }

    }

    ret=0;

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
