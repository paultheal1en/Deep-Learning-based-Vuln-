static void nbd_parse_filename(const char *filename, QDict *options,

                               Error **errp)

{

    char *file;

    char *export_name;

    const char *host_spec;

    const char *unixpath;



    if (nbd_has_filename_options_conflict(options, errp)) {

        return;

    }



    if (strstr(filename, "://")) {

        int ret = nbd_parse_uri(filename, options);

        if (ret < 0) {

            error_setg(errp, "No valid URL specified");

        }

        return;

    }



    file = g_strdup(filename);



    export_name = strstr(file, EN_OPTSTR);

    if (export_name) {

        if (export_name[strlen(EN_OPTSTR)] == 0) {

            goto out;

        }

        export_name[0] = 0; /* truncate 'file' */

        export_name += strlen(EN_OPTSTR);



        qdict_put(options, "export", qstring_from_str(export_name));

    }



    /* extract the host_spec - fail if it's not nbd:... */

    if (!strstart(file, "nbd:", &host_spec)) {

        error_setg(errp, "File name string for NBD must start with 'nbd:'");

        goto out;

    }



    if (!*host_spec) {

        goto out;

    }



    /* are we a UNIX or TCP socket? */

    if (strstart(host_spec, "unix:", &unixpath)) {

        qdict_put(options, "server.type", qstring_from_str("unix"));

        qdict_put(options, "server.data.path", qstring_from_str(unixpath));

    } else {

        InetSocketAddress *addr = NULL;



        addr = inet_parse(host_spec, errp);

        if (!addr) {

            goto out;

        }



        qdict_put(options, "server.type", qstring_from_str("inet"));

        qdict_put(options, "server.data.host", qstring_from_str(addr->host));

        qdict_put(options, "server.data.port", qstring_from_str(addr->port));

        qapi_free_InetSocketAddress(addr);

    }



out:

    g_free(file);

}
