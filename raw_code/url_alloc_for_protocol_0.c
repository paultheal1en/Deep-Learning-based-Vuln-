static int url_alloc_for_protocol(URLContext **puc, struct URLProtocol *up,

                                  const char *filename, int flags,

                                  const AVIOInterruptCB *int_cb)

{

    URLContext *uc;

    int err;



#if CONFIG_NETWORK

    if (up->flags & URL_PROTOCOL_FLAG_NETWORK && !ff_network_init())

        return AVERROR(EIO);

#endif

    uc = av_mallocz(sizeof(URLContext) + strlen(filename) + 1);

    if (!uc) {

        err = AVERROR(ENOMEM);

        goto fail;

    }

    uc->av_class = &ffurl_context_class;

    uc->filename = (char *)&uc[1];

    strcpy(uc->filename, filename);

    uc->prot            = up;

    uc->flags           = flags;

    uc->is_streamed     = 0; /* default = not streamed */

    uc->max_packet_size = 0; /* default: stream file */

    if (up->priv_data_size) {

        uc->priv_data = av_mallocz(up->priv_data_size);

        if (!uc->priv_data) {

            err = AVERROR(ENOMEM);

            goto fail;

        }

        if (up->priv_data_class) {

            *(const AVClass **)uc->priv_data = up->priv_data_class;

            av_opt_set_defaults(uc->priv_data);

        }

    }

    if (int_cb)

        uc->interrupt_callback = *int_cb;



    *puc = uc;

    return 0;

fail:

    *puc = NULL;

    if (uc)

        av_freep(&uc->priv_data);

    av_freep(&uc);

#if CONFIG_NETWORK

    if (up->flags & URL_PROTOCOL_FLAG_NETWORK)

        ff_network_close();

#endif

    return err;

}
