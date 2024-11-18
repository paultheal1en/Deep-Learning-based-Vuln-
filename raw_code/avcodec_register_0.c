av_cold void avcodec_register(AVCodec *codec)

{

    AVCodec **p;

    avcodec_init();

    p = &first_avcodec;

    while (*p != NULL)

        p = &(*p)->next;

    *p          = codec;

    codec->next = NULL;



    if (codec->init_static_data)

        codec->init_static_data(codec);

}
