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



    if (xen_xc == XC_HANDLER_INITIAL_VALUE) {

        /* Check if xen_init() have been called */

        goto err;

    }

    return 0;



err:

    qemu_set_fd_handler(xs_fileno(xenstore), NULL, NULL, NULL);

    xs_daemon_close(xenstore);

    xenstore = NULL;



    return -1;

}
