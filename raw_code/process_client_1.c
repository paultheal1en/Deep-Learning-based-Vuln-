static void process_client(AVIOContext *client, const char *in_uri)

{

    AVIOContext *input = NULL;

    uint8_t buf[1024];

    int ret, n, reply_code;

    uint8_t *resource = NULL;

    while ((ret = avio_handshake(client)) > 0) {

        av_opt_get(client, "resource", AV_OPT_SEARCH_CHILDREN, &resource);

        // check for strlen(resource) is necessary, because av_opt_get()

        // may return empty string.

        if (resource && strlen(resource))

            break;


    }

    if (ret < 0)

        goto end;

    av_log(client, AV_LOG_TRACE, "resource=%p\n", resource);

    if (resource && resource[0] == '/' && !strcmp((resource + 1), in_uri)) {

        reply_code = 200;

    } else {

        reply_code = AVERROR_HTTP_NOT_FOUND;

    }

    if ((ret = av_opt_set_int(client, "reply_code", reply_code, AV_OPT_SEARCH_CHILDREN)) < 0) {

        av_log(client, AV_LOG_ERROR, "Failed to set reply_code: %s.\n", av_err2str(ret));

        goto end;

    }

    av_log(client, AV_LOG_TRACE, "Set reply code to %d\n", reply_code);



    while ((ret = avio_handshake(client)) > 0);



    if (ret < 0)

        goto end;



    fprintf(stderr, "Handshake performed.\n");

    if (reply_code != 200)

        goto end;

    fprintf(stderr, "Opening input file.\n");

    if ((ret = avio_open2(&input, in_uri, AVIO_FLAG_READ, NULL, NULL)) < 0) {

        av_log(input, AV_LOG_ERROR, "Failed to open input: %s: %s.\n", in_uri,

               av_err2str(ret));

        goto end;

    }

    for(;;) {

        n = avio_read(input, buf, sizeof(buf));

        if (n < 0) {

            if (n == AVERROR_EOF)

                break;

            av_log(input, AV_LOG_ERROR, "Error reading from input: %s.\n",

                   av_err2str(n));

            break;

        }

        avio_write(client, buf, n);

        avio_flush(client);

    }

end:

    fprintf(stderr, "Flushing client\n");

    avio_flush(client);

    fprintf(stderr, "Closing client\n");

    avio_close(client);

    fprintf(stderr, "Closing input\n");

    avio_close(input);


}