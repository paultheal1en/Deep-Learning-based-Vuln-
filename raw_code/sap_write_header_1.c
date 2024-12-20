static int sap_write_header(AVFormatContext *s)

{

    struct SAPState *sap = s->priv_data;

    char host[1024], path[1024], url[1024], announce_addr[50] = "";

    char *option_list;

    int port = 9875, base_port = 5004, i, pos = 0, same_port = 0, ttl = 255;

    AVFormatContext **contexts = NULL;

    int ret = 0;

    struct sockaddr_storage localaddr;

    socklen_t addrlen = sizeof(localaddr);

    int udp_fd;

    AVDictionaryEntry* title = av_dict_get(s->metadata, "title", NULL, 0);



    if (!ff_network_init())

        return AVERROR(EIO);



    /* extract hostname and port */

    av_url_split(NULL, 0, NULL, 0, host, sizeof(host), &base_port,

                 path, sizeof(path), s->filename);

    if (base_port < 0)

        base_port = 5004;



    /* search for options */

    option_list = strrchr(path, '?');

    if (option_list) {

        char buf[50];

        if (av_find_info_tag(buf, sizeof(buf), "announce_port", option_list)) {

            port = strtol(buf, NULL, 10);

        }

        if (av_find_info_tag(buf, sizeof(buf), "same_port", option_list)) {

            same_port = strtol(buf, NULL, 10);

        }

        if (av_find_info_tag(buf, sizeof(buf), "ttl", option_list)) {

            ttl = strtol(buf, NULL, 10);

        }

        if (av_find_info_tag(buf, sizeof(buf), "announce_addr", option_list)) {

            av_strlcpy(announce_addr, buf, sizeof(announce_addr));

        }

    }



    if (!announce_addr[0]) {

        struct addrinfo hints = { 0 }, *ai = NULL;

        hints.ai_family = AF_UNSPEC;

        if (getaddrinfo(host, NULL, &hints, &ai)) {

            av_log(s, AV_LOG_ERROR, "Unable to resolve %s\n", host);

            ret = AVERROR(EIO);

            goto fail;

        }

        if (ai->ai_family == AF_INET) {

            /* Also known as sap.mcast.net */

            av_strlcpy(announce_addr, "224.2.127.254", sizeof(announce_addr));

#if HAVE_STRUCT_SOCKADDR_IN6

        } else if (ai->ai_family == AF_INET6) {

            /* With IPv6, you can use the same destination in many different

             * multicast subnets, to choose how far you want it routed.

             * This one is intended to be routed globally. */

            av_strlcpy(announce_addr, "ff0e::2:7ffe", sizeof(announce_addr));

#endif

        } else {

            freeaddrinfo(ai);

            av_log(s, AV_LOG_ERROR, "Host %s resolved to unsupported "

                                    "address family\n", host);

            ret = AVERROR(EIO);

            goto fail;

        }

        freeaddrinfo(ai);

    }



    sap->protocols = ffurl_get_protocols(NULL, NULL);

    if (!sap->protocols) {

        ret = AVERROR(ENOMEM);

        goto fail;

    }



    contexts = av_mallocz(sizeof(AVFormatContext*) * s->nb_streams);

    if (!contexts) {

        ret = AVERROR(ENOMEM);

        goto fail;

    }



    s->start_time_realtime = av_gettime();

    for (i = 0; i < s->nb_streams; i++) {

        URLContext *fd;



        ff_url_join(url, sizeof(url), "rtp", NULL, host, base_port,

                    "?ttl=%d", ttl);

        if (!same_port)

            base_port += 2;

        ret = ffurl_open(&fd, url, AVIO_FLAG_WRITE, &s->interrupt_callback, NULL,

                         sap->protocols);

        if (ret) {

            ret = AVERROR(EIO);

            goto fail;

        }

        ret = ff_rtp_chain_mux_open(&contexts[i], s, s->streams[i], fd, 0, i);

        if (ret < 0)

            goto fail;

        s->streams[i]->priv_data = contexts[i];

        s->streams[i]->time_base = contexts[i]->streams[0]->time_base;

        av_strlcpy(contexts[i]->filename, url, sizeof(contexts[i]->filename));

    }



    if (s->nb_streams > 0 && title)

        av_dict_set(&contexts[0]->metadata, "title", title->value, 0);



    ff_url_join(url, sizeof(url), "udp", NULL, announce_addr, port,

                "?ttl=%d&connect=1", ttl);

    ret = ffurl_open(&sap->ann_fd, url, AVIO_FLAG_WRITE,

                     &s->interrupt_callback, NULL, sap->protocols);

    if (ret) {

        ret = AVERROR(EIO);

        goto fail;

    }



    udp_fd = ffurl_get_file_handle(sap->ann_fd);

    if (getsockname(udp_fd, (struct sockaddr*) &localaddr, &addrlen)) {

        ret = AVERROR(EIO);

        goto fail;

    }

    if (localaddr.ss_family != AF_INET

#if HAVE_STRUCT_SOCKADDR_IN6

        && localaddr.ss_family != AF_INET6

#endif

        ) {

        av_log(s, AV_LOG_ERROR, "Unsupported protocol family\n");

        ret = AVERROR(EIO);

        goto fail;

    }

    sap->ann_size = 8192;

    sap->ann = av_mallocz(sap->ann_size);

    if (!sap->ann) {

        ret = AVERROR(EIO);

        goto fail;

    }

    sap->ann[pos] = (1 << 5);

#if HAVE_STRUCT_SOCKADDR_IN6

    if (localaddr.ss_family == AF_INET6)

        sap->ann[pos] |= 0x10;

#endif

    pos++;

    sap->ann[pos++] = 0; /* Authentication length */

    AV_WB16(&sap->ann[pos], av_get_random_seed());

    pos += 2;

    if (localaddr.ss_family == AF_INET) {

        memcpy(&sap->ann[pos], &((struct sockaddr_in*)&localaddr)->sin_addr,

               sizeof(struct in_addr));

        pos += sizeof(struct in_addr);

#if HAVE_STRUCT_SOCKADDR_IN6

    } else {

        memcpy(&sap->ann[pos], &((struct sockaddr_in6*)&localaddr)->sin6_addr,

               sizeof(struct in6_addr));

        pos += sizeof(struct in6_addr);

#endif

    }



    av_strlcpy(&sap->ann[pos], "application/sdp", sap->ann_size - pos);

    pos += strlen(&sap->ann[pos]) + 1;



    if (av_sdp_create(contexts, s->nb_streams, &sap->ann[pos],

                      sap->ann_size - pos)) {

        ret = AVERROR_INVALIDDATA;

        goto fail;

    }

    av_freep(&contexts);

    av_log(s, AV_LOG_VERBOSE, "SDP:\n%s\n", &sap->ann[pos]);

    pos += strlen(&sap->ann[pos]);

    sap->ann_size = pos;



    if (sap->ann_size > sap->ann_fd->max_packet_size) {

        av_log(s, AV_LOG_ERROR, "Announcement too large to send in one "

                                "packet\n");

        goto fail;

    }



    return 0;



fail:

    av_free(contexts);

    sap_write_close(s);

    return ret;

}
