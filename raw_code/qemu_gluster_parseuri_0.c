static int qemu_gluster_parseuri(GlusterConf *gconf, const char *filename)

{

    URI *uri;

    QueryParams *qp = NULL;

    bool is_unix = false;

    int ret = 0;



    uri = uri_parse(filename);

    if (!uri) {

        return -EINVAL;

    }



    /* transport */

    if (!strcmp(uri->scheme, "gluster")) {

        gconf->transport = g_strdup("tcp");

    } else if (!strcmp(uri->scheme, "gluster+tcp")) {

        gconf->transport = g_strdup("tcp");

    } else if (!strcmp(uri->scheme, "gluster+unix")) {

        gconf->transport = g_strdup("unix");

        is_unix = true;

    } else if (!strcmp(uri->scheme, "gluster+rdma")) {

        gconf->transport = g_strdup("rdma");

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

        gconf->server = g_strdup(qp->p[0].value);

    } else {

        gconf->server = g_strdup(uri->server);

        gconf->port = uri->port;

    }



out:

    if (qp) {

        query_params_free(qp);

    }

    uri_free(uri);

    return ret;

}
