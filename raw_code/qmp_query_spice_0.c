SpiceInfo *qmp_query_spice(Error **errp)

{

    QemuOpts *opts = QTAILQ_FIRST(&qemu_spice_opts.head);

    int port, tls_port;

    const char *addr;

    SpiceInfo *info;

    char version_string[20]; /* 12 = |255.255.255\0| is the max */



    info = g_malloc0(sizeof(*info));



    if (!spice_server || !opts) {

        info->enabled = false;

        return info;

    }



    info->enabled = true;

    info->migrated = spice_migration_completed;



    addr = qemu_opt_get(opts, "addr");

    port = qemu_opt_get_number(opts, "port", 0);

    tls_port = qemu_opt_get_number(opts, "tls-port", 0);



    info->has_auth = true;

    info->auth = g_strdup(auth);



    info->has_host = true;

    info->host = g_strdup(addr ? addr : "0.0.0.0");



    info->has_compiled_version = true;

    snprintf(version_string, sizeof(version_string), "%d.%d.%d",

             (SPICE_SERVER_VERSION & 0xff0000) >> 16,

             (SPICE_SERVER_VERSION & 0xff00) >> 8,

             SPICE_SERVER_VERSION & 0xff);

    info->compiled_version = g_strdup(version_string);



    if (port) {

        info->has_port = true;

        info->port = port;

    }

    if (tls_port) {

        info->has_tls_port = true;

        info->tls_port = tls_port;

    }



    info->mouse_mode = spice_server_is_server_mouse(spice_server) ?

                       SPICE_QUERY_MOUSE_MODE_SERVER :

                       SPICE_QUERY_MOUSE_MODE_CLIENT;



    /* for compatibility with the original command */

    info->has_channels = true;

    info->channels = qmp_query_spice_channels();



    return info;

}
