static void nbd_refresh_filename(BlockDriverState *bs)

{

    QDict *opts = qdict_new();

    const char *path   = qdict_get_try_str(bs->options, "path");

    const char *host   = qdict_get_try_str(bs->options, "host");

    const char *port   = qdict_get_try_str(bs->options, "port");

    const char *export = qdict_get_try_str(bs->options, "export");



    qdict_put_obj(opts, "driver", QOBJECT(qstring_from_str("nbd")));



    if (path && export) {

        snprintf(bs->exact_filename, sizeof(bs->exact_filename),

                 "nbd+unix:///%s?socket=%s", export, path);

    } else if (path && !export) {

        snprintf(bs->exact_filename, sizeof(bs->exact_filename),

                 "nbd+unix://?socket=%s", path);

    } else if (!path && export && port) {

        snprintf(bs->exact_filename, sizeof(bs->exact_filename),

                 "nbd://%s:%s/%s", host, port, export);

    } else if (!path && export && !port) {

        snprintf(bs->exact_filename, sizeof(bs->exact_filename),

                 "nbd://%s/%s", host, export);

    } else if (!path && !export && port) {

        snprintf(bs->exact_filename, sizeof(bs->exact_filename),

                 "nbd://%s:%s", host, port);

    } else if (!path && !export && !port) {

        snprintf(bs->exact_filename, sizeof(bs->exact_filename),

                 "nbd://%s", host);

    }



    if (path) {

        qdict_put_obj(opts, "path", QOBJECT(qstring_from_str(path)));

    } else if (port) {

        qdict_put_obj(opts, "host", QOBJECT(qstring_from_str(host)));

        qdict_put_obj(opts, "port", QOBJECT(qstring_from_str(port)));

    } else {

        qdict_put_obj(opts, "host", QOBJECT(qstring_from_str(host)));

    }

    if (export) {

        qdict_put_obj(opts, "export", QOBJECT(qstring_from_str(export)));

    }



    bs->full_open_options = opts;

}
