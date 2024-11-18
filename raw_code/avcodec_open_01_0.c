int avcodec_open(AVCodecContext *avctx, AVCodec *codec)

{

    int ret;



    if(avctx->codec)

        return -1;



    avctx->codec = codec;

    avctx->codec_id = codec->id;

    avctx->frame_number = 0;

    if (codec->priv_data_size > 0) {

        avctx->priv_data = av_mallocz(codec->priv_data_size);

        if (!avctx->priv_data) 

            return -ENOMEM;

    } else {

        avctx->priv_data = NULL;

    }



    if(avctx->coded_width && avctx->coded_height)

        avcodec_set_dimensions(avctx, avctx->coded_width, avctx->coded_height);

    else if(avctx->width && avctx->height)

        avcodec_set_dimensions(avctx, avctx->width, avctx->height);



    if((avctx->coded_width||avctx->coded_height) && avcodec_check_dimensions(avctx,avctx->coded_width,avctx->coded_height)){

        av_freep(&avctx->priv_data);

        return -1;

    }



    ret = avctx->codec->init(avctx);

    if (ret < 0) {

        av_freep(&avctx->priv_data);

        return ret;

    }

    return 0;

}
