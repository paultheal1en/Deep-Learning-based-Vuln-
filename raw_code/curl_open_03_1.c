static int curl_open(BlockDriverState *bs, QDict *options, int flags)

{

    BDRVCURLState *s = bs->opaque;

    CURLState *state = NULL;

    QemuOpts *opts;

    Error *local_err = NULL;

    const char *file;

    double d;



    static int inited = 0;



    if (flags & BDRV_O_RDWR) {

        qerror_report(ERROR_CLASS_GENERIC_ERROR,

                      "curl block device does not support writes");

        return -EROFS;

    }



    opts = qemu_opts_create_nofail(&runtime_opts);

    qemu_opts_absorb_qdict(opts, options, &local_err);

    if (error_is_set(&local_err)) {

        qerror_report_err(local_err);

        error_free(local_err);

        goto out_noclean;

    }



    s->readahead_size = qemu_opt_get_size(opts, "readahead", READ_AHEAD_SIZE);

    if ((s->readahead_size & 0x1ff) != 0) {

        fprintf(stderr, "HTTP_READAHEAD_SIZE %zd is not a multiple of 512\n",

                s->readahead_size);

        goto out_noclean;

    }



    file = qemu_opt_get(opts, "url");

    if (file == NULL) {

        qerror_report(ERROR_CLASS_GENERIC_ERROR, "curl block driver requires "

                      "an 'url' option");

        goto out_noclean;

    }



    if (!inited) {

        curl_global_init(CURL_GLOBAL_ALL);

        inited = 1;

    }



    DPRINTF("CURL: Opening %s\n", file);

    s->url = g_strdup(file);

    state = curl_init_state(s);

    if (!state)

        goto out_noclean;



    // Get file size



    curl_easy_setopt(state->curl, CURLOPT_NOBODY, 1);

    curl_easy_setopt(state->curl, CURLOPT_WRITEFUNCTION, (void *)curl_size_cb);

    if (curl_easy_perform(state->curl))

        goto out;

    curl_easy_getinfo(state->curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD, &d);

    if (d)

        s->len = (size_t)d;

    else if(!s->len)

        goto out;

    DPRINTF("CURL: Size = %zd\n", s->len);



    curl_clean_state(state);

    curl_easy_cleanup(state->curl);

    state->curl = NULL;



    // Now we know the file exists and its size, so let's

    // initialize the multi interface!



    s->multi = curl_multi_init();

    curl_multi_setopt(s->multi, CURLMOPT_SOCKETDATA, s);

    curl_multi_setopt(s->multi, CURLMOPT_SOCKETFUNCTION, curl_sock_cb);

    curl_multi_do(s);



    qemu_opts_del(opts);

    return 0;



out:

    fprintf(stderr, "CURL: Error opening file: %s\n", state->errmsg);

    curl_easy_cleanup(state->curl);

    state->curl = NULL;

out_noclean:

    g_free(s->url);

    qemu_opts_del(opts);

    return -EINVAL;

}
