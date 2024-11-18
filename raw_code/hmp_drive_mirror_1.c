void hmp_drive_mirror(Monitor *mon, const QDict *qdict)

{

    const char *device = qdict_get_str(qdict, "device");

    const char *filename = qdict_get_str(qdict, "target");

    const char *format = qdict_get_try_str(qdict, "format");

    int reuse = qdict_get_try_bool(qdict, "reuse", 0);

    int full = qdict_get_try_bool(qdict, "full", 0);

    enum NewImageMode mode;

    Error *errp = NULL;



    if (!filename) {

        error_set(&errp, QERR_MISSING_PARAMETER, "target");

        hmp_handle_error(mon, &errp);

        return;

    }



    if (reuse) {

        mode = NEW_IMAGE_MODE_EXISTING;

    } else {

        mode = NEW_IMAGE_MODE_ABSOLUTE_PATHS;

    }



    qmp_drive_mirror(device, filename, !!format, format,

                     full ? MIRROR_SYNC_MODE_FULL : MIRROR_SYNC_MODE_TOP,

                     true, mode, false, 0, &errp);

    hmp_handle_error(mon, &errp);

}
