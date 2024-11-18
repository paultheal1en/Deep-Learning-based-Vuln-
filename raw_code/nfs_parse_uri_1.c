static int nfs_parse_uri(const char *filename, QDict *options, Error **errp)

{

    URI *uri = NULL;

    QueryParams *qp = NULL;

    int ret = -EINVAL, i;



    uri = uri_parse(filename);

    if (!uri) {

        error_setg(errp, "Invalid URI specified");

        goto out;

    }

    if (strcmp(uri->scheme, "nfs") != 0) {

        error_setg(errp, "URI scheme must be 'nfs'");

        goto out;

    }



    if (!uri->server) {

        error_setg(errp, "missing hostname in URI");

        goto out;

    }



    if (!uri->path) {

        error_setg(errp, "missing file path in URI");

        goto out;

    }



    qp = query_params_parse(uri->query);

    if (!qp) {

        error_setg(errp, "could not parse query parameters");

        goto out;

    }



    qdict_put(options, "server.host", qstring_from_str(uri->server));

    qdict_put(options, "server.type", qstring_from_str("inet"));

    qdict_put(options, "path", qstring_from_str(uri->path));



    for (i = 0; i < qp->n; i++) {

        if (!qp->p[i].value) {

            error_setg(errp, "Value for NFS parameter expected: %s",

                       qp->p[i].name);

            goto out;

        }

        if (parse_uint_full(qp->p[i].value, NULL, 0)) {

            error_setg(errp, "Illegal value for NFS parameter: %s",

                       qp->p[i].name);

            goto out;

        }

        if (!strcmp(qp->p[i].name, "uid")) {

            qdict_put(options, "user",

                      qstring_from_str(qp->p[i].value));

        } else if (!strcmp(qp->p[i].name, "gid")) {

            qdict_put(options, "group",

                      qstring_from_str(qp->p[i].value));

        } else if (!strcmp(qp->p[i].name, "tcp-syncnt")) {

            qdict_put(options, "tcp-syn-count",

                      qstring_from_str(qp->p[i].value));

        } else if (!strcmp(qp->p[i].name, "readahead")) {

            qdict_put(options, "readahead-size",

                      qstring_from_str(qp->p[i].value));

        } else if (!strcmp(qp->p[i].name, "pagecache")) {

            qdict_put(options, "page-cache-size",

                      qstring_from_str(qp->p[i].value));

        } else if (!strcmp(qp->p[i].name, "debug")) {

            qdict_put(options, "debug",

                      qstring_from_str(qp->p[i].value));

        } else {

            error_setg(errp, "Unknown NFS parameter name: %s",

                       qp->p[i].name);

            goto out;

        }

    }

    ret = 0;

out:

    if (qp) {

        query_params_free(qp);

    }

    if (uri) {

        uri_free(uri);

    }

    return ret;

}
