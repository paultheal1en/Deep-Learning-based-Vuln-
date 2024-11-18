int xen_be_init(void)

{

    xenstore = xs_daemon_open();

    if (!xenstore) {

        xen_be_printf(NULL, 0, "can't connect to xenstored\n");

        return -1;

    }



    if (qemu_set_fd_handler(xs_fileno(xenstore), xenstore_update, NULL, NULL) < 0) {

        goto err;

    }



    xen_xc = xc_interface_open();

    if (xen_xc == -1) {

        xen_be_printf(NULL, 0, "can't open xen interface\n");

        goto err;

    }

    return 0;



err:

    qemu_set_fd_handler(xs_fileno(xenstore), NULL, NULL, NULL);

    xs_daemon_close(xenstore);

    xenstore = NULL;



    return -1;

}
