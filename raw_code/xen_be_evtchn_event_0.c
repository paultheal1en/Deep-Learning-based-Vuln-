static void xen_be_evtchn_event(void *opaque)

{

    struct XenDevice *xendev = opaque;

    evtchn_port_t port;



    port = xc_evtchn_pending(xendev->evtchndev);

    if (port != xendev->local_port) {

        xen_be_printf(xendev, 0, "xc_evtchn_pending returned %d (expected %d)\n",

                      port, xendev->local_port);

        return;

    }

    xc_evtchn_unmask(xendev->evtchndev, port);



    if (xendev->ops->event) {

        xendev->ops->event(xendev);

    }

}
