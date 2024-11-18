static int qemu_gluster_parse_json(BlockdevOptionsGluster *gconf,

                                  QDict *options, Error **errp)

{

    QemuOpts *opts;

    SocketAddress *gsconf = NULL;

    SocketAddressList *curr = NULL;

    QDict *backing_options = NULL;

    Error *local_err = NULL;

    char *str = NULL;

    const char *ptr;

    size_t num_servers;

    int i, type;



    /* create opts info from runtime_json_opts list */

    opts = qemu_opts_create(&runtime_json_opts, NULL, 0, &error_abort);

    qemu_opts_absorb_qdict(opts, options, &local_err);

    if (local_err) {

        goto out;

    }



    num_servers = qdict_array_entries(options, GLUSTER_OPT_SERVER_PATTERN);

    if (num_servers < 1) {

        error_setg(&local_err, QERR_MISSING_PARAMETER, "server");

        goto out;

    }



    ptr = qemu_opt_get(opts, GLUSTER_OPT_VOLUME);

    if (!ptr) {

        error_setg(&local_err, QERR_MISSING_PARAMETER, GLUSTER_OPT_VOLUME);

        goto out;

    }

    gconf->volume = g_strdup(ptr);



    ptr = qemu_opt_get(opts, GLUSTER_OPT_PATH);

    if (!ptr) {

        error_setg(&local_err, QERR_MISSING_PARAMETER, GLUSTER_OPT_PATH);

        goto out;

    }

    gconf->path = g_strdup(ptr);

    qemu_opts_del(opts);



    for (i = 0; i < num_servers; i++) {

        str = g_strdup_printf(GLUSTER_OPT_SERVER_PATTERN"%d.", i);

        qdict_extract_subqdict(options, &backing_options, str);



        /* create opts info from runtime_type_opts list */

        opts = qemu_opts_create(&runtime_type_opts, NULL, 0, &error_abort);

        qemu_opts_absorb_qdict(opts, backing_options, &local_err);

        if (local_err) {

            goto out;

        }



        ptr = qemu_opt_get(opts, GLUSTER_OPT_TYPE);

        if (!ptr) {

            error_setg(&local_err, QERR_MISSING_PARAMETER, GLUSTER_OPT_TYPE);

            error_append_hint(&local_err, GERR_INDEX_HINT, i);

            goto out;



        }

        gsconf = g_new0(SocketAddress, 1);

        if (!strcmp(ptr, "tcp")) {

            ptr = "inet";       /* accept legacy "tcp" */

        }

        type = qapi_enum_parse(SocketAddressType_lookup, ptr,

                               SOCKET_ADDRESS_TYPE__MAX, -1, NULL);

        if (type != SOCKET_ADDRESS_TYPE_INET

            && type != SOCKET_ADDRESS_TYPE_UNIX) {

            error_setg(&local_err,

                       "Parameter '%s' may be 'inet' or 'unix'",

                       GLUSTER_OPT_TYPE);

            error_append_hint(&local_err, GERR_INDEX_HINT, i);

            goto out;

        }

        gsconf->type = type;

        qemu_opts_del(opts);



        if (gsconf->type == SOCKET_ADDRESS_TYPE_INET) {

            /* create opts info from runtime_inet_opts list */

            opts = qemu_opts_create(&runtime_inet_opts, NULL, 0, &error_abort);

            qemu_opts_absorb_qdict(opts, backing_options, &local_err);

            if (local_err) {

                goto out;

            }



            ptr = qemu_opt_get(opts, GLUSTER_OPT_HOST);

            if (!ptr) {

                error_setg(&local_err, QERR_MISSING_PARAMETER,

                           GLUSTER_OPT_HOST);

                error_append_hint(&local_err, GERR_INDEX_HINT, i);

                goto out;

            }

            gsconf->u.inet.host = g_strdup(ptr);

            ptr = qemu_opt_get(opts, GLUSTER_OPT_PORT);

            if (!ptr) {

                error_setg(&local_err, QERR_MISSING_PARAMETER,

                           GLUSTER_OPT_PORT);

                error_append_hint(&local_err, GERR_INDEX_HINT, i);

                goto out;

            }

            gsconf->u.inet.port = g_strdup(ptr);



            /* defend for unsupported fields in InetSocketAddress,

             * i.e. @ipv4, @ipv6  and @to

             */

            ptr = qemu_opt_get(opts, GLUSTER_OPT_TO);

            if (ptr) {

                gsconf->u.inet.has_to = true;

            }

            ptr = qemu_opt_get(opts, GLUSTER_OPT_IPV4);

            if (ptr) {

                gsconf->u.inet.has_ipv4 = true;

            }

            ptr = qemu_opt_get(opts, GLUSTER_OPT_IPV6);

            if (ptr) {

                gsconf->u.inet.has_ipv6 = true;

            }

            if (gsconf->u.inet.has_to) {

                error_setg(&local_err, "Parameter 'to' not supported");

                goto out;

            }

            if (gsconf->u.inet.has_ipv4 || gsconf->u.inet.has_ipv6) {

                error_setg(&local_err, "Parameters 'ipv4/ipv6' not supported");

                goto out;

            }

            qemu_opts_del(opts);

        } else {

            /* create opts info from runtime_unix_opts list */

            opts = qemu_opts_create(&runtime_unix_opts, NULL, 0, &error_abort);

            qemu_opts_absorb_qdict(opts, backing_options, &local_err);

            if (local_err) {

                goto out;

            }



            ptr = qemu_opt_get(opts, GLUSTER_OPT_SOCKET);

            if (!ptr) {

                error_setg(&local_err, QERR_MISSING_PARAMETER,

                           GLUSTER_OPT_SOCKET);

                error_append_hint(&local_err, GERR_INDEX_HINT, i);

                goto out;

            }

            gsconf->u.q_unix.path = g_strdup(ptr);

            qemu_opts_del(opts);

        }



        if (gconf->server == NULL) {

            gconf->server = g_new0(SocketAddressList, 1);

            gconf->server->value = gsconf;

            curr = gconf->server;

        } else {

            curr->next = g_new0(SocketAddressList, 1);

            curr->next->value = gsconf;

            curr = curr->next;

        }

        gsconf = NULL;



        QDECREF(backing_options);

        backing_options = NULL;

        g_free(str);

        str = NULL;

    }



    return 0;



out:

    error_propagate(errp, local_err);

    qapi_free_SocketAddress(gsconf);

    qemu_opts_del(opts);

    g_free(str);

    QDECREF(backing_options);

    errno = EINVAL;

    return -errno;

}
