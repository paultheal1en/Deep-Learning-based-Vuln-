static int url_alloc_for_protocol (URLContext **puc, struct URLProtocol *up,

                                   const char *filename, int flags,

                                   const AVIOInterruptCB *int_cb)

{

    URLContext *uc;

    int err;



#if CONFIG_NETWORK

    if (up->flags & URL_PROTOCOL_FLAG_NETWORK && !ff_network_init())

        return AVERROR(EIO);

#endif

    if ((flags & AVIO_FLAG_READ) && !up->url_read) {

        av_log(NULL, AV_LOG_ERROR,

               "Impossible to open the '%s' protocol for reading\n", up->name);

        return AVERROR(EIO);


    if ((flags & AVIO_FLAG_WRITE) && !up->url_write) {

        av_log(NULL, AV_LOG_ERROR,

               "Impossible to open the '%s' protocol for writing\n", up->name);

        return AVERROR(EIO);


    uc = av_mallocz(sizeof(URLContext) + strlen(filename) + 1);

    if (!uc) {




    uc->av_class = &ffurl_context_class;

    uc->filename = (char *) &uc[1];

    strcpy(uc->filename, filename);

    uc->prot = up;

    uc->flags = flags;

    uc->is_streamed = 0; /* default = not streamed */

    uc->max_packet_size = 0; /* default: stream file */

    if (up->priv_data_size) {

        uc->priv_data = av_mallocz(up->priv_data_size);





        if (up->priv_data_class) {

            int proto_len= strlen(up->name);

            char *start = strchr(uc->filename, ',');

            *(const AVClass**)uc->priv_data = up->priv_data_class;

            av_opt_set_defaults(uc->priv_data);

            if(!strncmp(up->name, uc->filename, proto_len) && uc->filename + proto_len == start){

                int ret= 0;

                char *p= start;

                char sep= *++p;

                char *key, *val;

                p++;

                while(ret >= 0 && (key= strchr(p, sep)) && p<key && (val = strchr(key+1, sep))){

                    *val= *key= 0;

                    ret= av_opt_set(uc->priv_data, p, key+1, 0);

                    if (ret == AVERROR_OPTION_NOT_FOUND)

                        av_log(uc, AV_LOG_ERROR, "Key '%s' not found.\n", p);

                    *val= *key= sep;

                    p= val+1;


                if(ret<0 || p!=key){

                    av_log(uc, AV_LOG_ERROR, "Error parsing options string %s\n", start);

                    av_freep(&uc->priv_data);

                    av_freep(&uc);

                    err = AVERROR(EINVAL);



                memmove(start, key+1, strlen(key));




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
