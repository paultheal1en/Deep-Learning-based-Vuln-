void hmp_change(Monitor *mon, const QDict *qdict)

{

    const char *device = qdict_get_str(qdict, "device");

    const char *target = qdict_get_str(qdict, "target");

    const char *arg = qdict_get_try_str(qdict, "arg");

    Error *err = NULL;



    if (strcmp(device, "vnc") == 0 &&

            (strcmp(target, "passwd") == 0 ||

             strcmp(target, "password") == 0)) {

        if (!arg) {

            monitor_read_password(mon, hmp_change_read_arg, NULL);

            return;

        }

    }



    qmp_change(device, target, !!arg, arg, &err);

    if (error_is_type(err, QERR_DEVICE_ENCRYPTED)) {

        monitor_printf(mon, "%s (%s) is encrypted.\n",

                       error_get_field(err, "device"),

                       error_get_field(err, "filename"));

        if (!monitor_get_rs(mon)) {

            monitor_printf(mon,

                    "terminal does not support password prompting\n");

            error_free(err);

            return;

        }

        readline_start(monitor_get_rs(mon), "Password: ", 1,

                       cb_hmp_change_bdrv_pwd, err);

        return;

    }

    hmp_handle_error(mon, &err);

}
