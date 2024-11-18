static int add_graphics_client(Monitor *mon, const QDict *qdict, QObject **ret_data)

{

    const char *protocol  = qdict_get_str(qdict, "protocol");

    const char *fdname = qdict_get_str(qdict, "fdname");

    CharDriverState *s;



    if (strcmp(protocol, "spice") == 0) {

        int fd = monitor_get_fd(mon, fdname, NULL);

        int skipauth = qdict_get_try_bool(qdict, "skipauth", 0);

        int tls = qdict_get_try_bool(qdict, "tls", 0);

        if (!using_spice) {

            /* correct one? spice isn't a device ,,, */

            qerror_report(QERR_DEVICE_NOT_ACTIVE, "spice");

            return -1;

        }

        if (qemu_spice_display_add_client(fd, skipauth, tls) < 0) {

            close(fd);

        }

        return 0;

#ifdef CONFIG_VNC

    } else if (strcmp(protocol, "vnc") == 0) {

	int fd = monitor_get_fd(mon, fdname, NULL);

        int skipauth = qdict_get_try_bool(qdict, "skipauth", 0);

	vnc_display_add_client(NULL, fd, skipauth);

	return 0;

#endif

    } else if ((s = qemu_chr_find(protocol)) != NULL) {

	int fd = monitor_get_fd(mon, fdname, NULL);

	if (qemu_chr_add_client(s, fd) < 0) {

	    qerror_report(QERR_ADD_CLIENT_FAILED);

	    return -1;

	}

	return 0;

    }



    qerror_report(QERR_INVALID_PARAMETER, "protocol");

    return -1;

}
