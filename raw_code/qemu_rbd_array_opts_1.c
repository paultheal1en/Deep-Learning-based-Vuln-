static char *qemu_rbd_array_opts(QDict *options, const char *prefix, int type,

                                 Error **errp)

{

    int num_entries;

    QemuOpts *opts = NULL;

    QDict *sub_options;

    const char *host;

    const char *port;

    char *str;

    char *rados_str = NULL;

    Error *local_err = NULL;

    int i;



    assert(type == RBD_MON_HOST);



    num_entries = qdict_array_entries(options, prefix);



    if (num_entries < 0) {

        error_setg(errp, "Parse error on RBD QDict array");

        return NULL;

    }



    for (i = 0; i < num_entries; i++) {

        char *strbuf = NULL;

        const char *value;

        char *rados_str_tmp;



        str = g_strdup_printf("%s%d.", prefix, i);

        qdict_extract_subqdict(options, &sub_options, str);

        g_free(str);



        opts = qemu_opts_create(&runtime_opts, NULL, 0, &error_abort);

        qemu_opts_absorb_qdict(opts, sub_options, &local_err);

        QDECREF(sub_options);

        if (local_err) {

            error_propagate(errp, local_err);

            g_free(rados_str);

            rados_str = NULL;

            goto exit;

        }



        if (type == RBD_MON_HOST) {

            host = qemu_opt_get(opts, "host");

            port = qemu_opt_get(opts, "port");



            value = host;

            if (port) {

                /* check for ipv6 */

                if (strchr(host, ':')) {

                    strbuf = g_strdup_printf("[%s]:%s", host, port);

                } else {

                    strbuf = g_strdup_printf("%s:%s", host, port);

                }

                value = strbuf;

            } else if (strchr(host, ':')) {

                strbuf = g_strdup_printf("[%s]", host);

                value = strbuf;

            }

        } else {

            abort();

        }



        /* each iteration in the for loop will build upon the string, and if

         * rados_str is NULL then it is our first pass */

        if (rados_str) {

            /* separate options with ';', as that  is what rados_conf_set()

             * requires */

            rados_str_tmp = rados_str;

            rados_str = g_strdup_printf("%s;%s", rados_str_tmp, value);

            g_free(rados_str_tmp);

        } else {

            rados_str = g_strdup(value);

        }



        g_free(strbuf);

        qemu_opts_del(opts);

        opts = NULL;

    }



exit:

    qemu_opts_del(opts);

    return rados_str;

}
