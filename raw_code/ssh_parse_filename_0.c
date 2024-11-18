static void ssh_parse_filename(const char *filename, QDict *options,

                               Error **errp)

{

    if (qdict_haskey(options, "user") ||

        qdict_haskey(options, "host") ||

        qdict_haskey(options, "port") ||

        qdict_haskey(options, "path") ||

        qdict_haskey(options, "host_key_check")) {

        error_setg(errp, "user, host, port, path, host_key_check cannot be used at the same time as a file option");

        return;

    }



    parse_uri(filename, options, errp);

}
