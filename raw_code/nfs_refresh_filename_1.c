static void nfs_refresh_filename(BlockDriverState *bs, QDict *options)

{

    NFSClient *client = bs->opaque;

    QDict *opts = qdict_new();

    QObject *server_qdict;

    Visitor *ov;



    qdict_put(opts, "driver", qstring_from_str("nfs"));



    if (client->uid && !client->gid) {

        snprintf(bs->exact_filename, sizeof(bs->exact_filename),

                 "nfs://%s%s?uid=%" PRId64, client->server->host, client->path,

                 client->uid);

    } else if (!client->uid && client->gid) {

        snprintf(bs->exact_filename, sizeof(bs->exact_filename),

                 "nfs://%s%s?gid=%" PRId64, client->server->host, client->path,

                 client->gid);

    } else if (client->uid && client->gid) {

        snprintf(bs->exact_filename, sizeof(bs->exact_filename),

                 "nfs://%s%s?uid=%" PRId64 "&gid=%" PRId64,

                 client->server->host, client->path, client->uid, client->gid);

    } else {

        snprintf(bs->exact_filename, sizeof(bs->exact_filename),

                 "nfs://%s%s", client->server->host, client->path);

    }



    ov = qobject_output_visitor_new(&server_qdict);

    visit_type_NFSServer(ov, NULL, &client->server, &error_abort);

    visit_complete(ov, &server_qdict);

    assert(qobject_type(server_qdict) == QTYPE_QDICT);



    qdict_put_obj(opts, "server", server_qdict);

    qdict_put(opts, "path", qstring_from_str(client->path));



    if (client->uid) {

        qdict_put(opts, "uid", qint_from_int(client->uid));

    }

    if (client->gid) {

        qdict_put(opts, "gid", qint_from_int(client->gid));

    }

    if (client->tcp_syncnt) {

        qdict_put(opts, "tcp-syncnt",

                      qint_from_int(client->tcp_syncnt));

    }

    if (client->readahead) {

        qdict_put(opts, "readahead",

                      qint_from_int(client->readahead));

    }

    if (client->pagecache) {

        qdict_put(opts, "pagecache",

                      qint_from_int(client->pagecache));

    }

    if (client->debug) {

        qdict_put(opts, "debug", qint_from_int(client->debug));

    }



    visit_free(ov);

    qdict_flatten(opts);

    bs->full_open_options = opts;

}
