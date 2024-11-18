static int con_init(struct XenDevice *xendev)

{

    struct XenConsole *con = container_of(xendev, struct XenConsole, xendev);

    char *type, *dom;

    int ret = 0;



    /* setup */

    dom = xs_get_domain_path(xenstore, con->xendev.dom);

    snprintf(con->console, sizeof(con->console), "%s/console", dom);

    free(dom);



    type = xenstore_read_str(con->console, "type");

    if (!type || strcmp(type, "ioemu") != 0) {

	xen_be_printf(xendev, 1, "not for me (type=%s)\n", type);

        ret = -1;

        goto out;

    }



    if (!serial_hds[con->xendev.dev])

	xen_be_printf(xendev, 1, "WARNING: serial line %d not configured\n",

                      con->xendev.dev);

    else

        con->chr = serial_hds[con->xendev.dev];



out:

    qemu_free(type);

    return ret;

}
