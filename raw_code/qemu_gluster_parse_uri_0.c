static int qemu_gluster_parse_uri(BlockdevOptionsGluster *gconf,

                                  const char *filename)

{

    SocketAddress *gsconf;

    URI *uri;

    QueryParams *qp = NULL;

    bool is_unix = false;

    int ret = 0;



    uri = uri_parse(filename);

    if (!uri) {

        return -EINVAL;

    }



    gconf->server = g_new0(SocketAddressList, 1);

    gconf->server->value = gsconf = g_new0(SocketAddress, 1);



    /* transport */

    if (!uri->scheme || !strcmp(uri->scheme, "gluster")) {

        gsconf->type = SOCKET_ADDRESS_TYPE_INET;

    } else if (!strcmp(uri->scheme, "gluster+tcp")) {

        gsconf->type = SOCKET_ADDRESS_TYPE_INET;

    } else if (!strcmp(uri->scheme, "gluster+unix")) {

        gsconf->type = SOCKET_ADDRESS_TYPE_UNIX;

        is_unix = true;

    } else if (!strcmp(uri->scheme, "gluster+rdma")) {

        gsconf->type = SOCKET_ADDRESS_TYPE_INET;

        error_report("Warning: rdma feature is not supported, falling "

                     "back to tcp");

    } else {

        ret = -EINVAL;

        goto out;

    }



    ret = parse_volume_options(gconf, uri->path);

    if (ret < 0) {

        goto out;

    }



    qp = query_params_parse(uri->query);

    if (qp->n > 1 || (is_unix && !qp->n) || (!is_unix && qp->n)) {

        ret = -EINVAL;

        goto out;

    }



    if (is_unix) {

        if (uri->server || uri->port) {

            ret = -EINVAL;

            goto out;

        }

        if (strcmp(qp->p[0].name, "socket")) {

            ret = -EINVAL;

            goto out;

        }

        gsconf->u.q_unix.path = g_strdup(qp->p[0].value);

    } else {

        gsconf->u.inet.host = g_strdup(uri->server ? uri->server : "localhost");

        if (uri->port) {

            gsconf->u.inet.port = g_strdup_printf("%d", uri->port);

        } else {

            gsconf->u.inet.port = g_strdup_printf("%d", GLUSTER_DEFAULT_PORT);

        }

    }



out:

    if (qp) {

        query_params_free(qp);

    }

    uri_free(uri);

    return ret;

}
