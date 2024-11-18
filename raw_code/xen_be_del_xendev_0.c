static struct XenDevice *xen_be_del_xendev(int dom, int dev)

{

    struct XenDevice *xendev, *xnext;



    /*

     * This is pretty much like QTAILQ_FOREACH(xendev, &xendevs, next) but

     * we save the next pointer in xnext because we might free xendev.

     */

    xnext = xendevs.tqh_first;

    while (xnext) {

        xendev = xnext;

        xnext = xendev->next.tqe_next;



        if (xendev->dom != dom) {

            continue;

        }

        if (xendev->dev != dev && dev != -1) {

            continue;

        }



        if (xendev->ops->free) {

            xendev->ops->free(xendev);

        }



        if (xendev->fe) {

            char token[XEN_BUFSIZE];

            snprintf(token, sizeof(token), "fe:%p", xendev);

            xs_unwatch(xenstore, xendev->fe, token);

            g_free(xendev->fe);

        }



        if (xendev->evtchndev != XC_HANDLER_INITIAL_VALUE) {

            xc_evtchn_close(xendev->evtchndev);

        }

        if (xendev->gnttabdev != XC_HANDLER_INITIAL_VALUE) {

            xc_gnttab_close(xendev->gnttabdev);

        }



        QTAILQ_REMOVE(&xendevs, xendev, next);

        g_free(xendev);

    }

    return NULL;

}
