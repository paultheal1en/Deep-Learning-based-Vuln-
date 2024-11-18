static struct XenDevice *xen_be_get_xendev(const char *type, int dom, int dev,

                                           struct XenDevOps *ops)

{

    struct XenDevice *xendev;

    char *dom0;



    xendev = xen_be_find_xendev(type, dom, dev);

    if (xendev) {

        return xendev;

    }



    /* init new xendev */

    xendev = g_malloc0(ops->size);

    xendev->type  = type;

    xendev->dom   = dom;

    xendev->dev   = dev;

    xendev->ops   = ops;



    dom0 = xs_get_domain_path(xenstore, 0);

    snprintf(xendev->be, sizeof(xendev->be), "%s/backend/%s/%d/%d",

             dom0, xendev->type, xendev->dom, xendev->dev);

    snprintf(xendev->name, sizeof(xendev->name), "%s-%d",

             xendev->type, xendev->dev);

    free(dom0);



    xendev->debug      = debug;

    xendev->local_port = -1;



    xendev->evtchndev = xen_xc_evtchn_open(NULL, 0);

    if (xendev->evtchndev == XC_HANDLER_INITIAL_VALUE) {

        xen_be_printf(NULL, 0, "can't open evtchn device\n");

        g_free(xendev);

        return NULL;

    }

    fcntl(xc_evtchn_fd(xendev->evtchndev), F_SETFD, FD_CLOEXEC);



    if (ops->flags & DEVOPS_FLAG_NEED_GNTDEV) {

        xendev->gnttabdev = xen_xc_gnttab_open(NULL, 0);

        if (xendev->gnttabdev == XC_HANDLER_INITIAL_VALUE) {

            xen_be_printf(NULL, 0, "can't open gnttab device\n");

            xc_evtchn_close(xendev->evtchndev);

            g_free(xendev);

            return NULL;

        }

    } else {

        xendev->gnttabdev = XC_HANDLER_INITIAL_VALUE;

    }



    QTAILQ_INSERT_TAIL(&xendevs, xendev, next);



    if (xendev->ops->alloc) {

        xendev->ops->alloc(xendev);

    }



    return xendev;

}
