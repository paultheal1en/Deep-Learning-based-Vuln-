static void con_disconnect(struct XenDevice *xendev)

{

    struct XenConsole *con = container_of(xendev, struct XenConsole, xendev);



    if (!xendev->dev) {

        return;

    }

    if (con->chr) {

        qemu_chr_add_handlers(con->chr, NULL, NULL, NULL, NULL);

        qemu_chr_fe_release(con->chr);

    }

    xen_be_unbind_evtchn(&con->xendev);



    if (con->sring) {

        if (!xendev->gnttabdev) {

            munmap(con->sring, XC_PAGE_SIZE);

        } else {

            xc_gnttab_munmap(xendev->gnttabdev, con->sring, 1);

        }

	con->sring = NULL;

    }

}
