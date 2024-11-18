int xen_be_send_notify(struct XenDevice *xendev)

{

    return xc_evtchn_notify(xendev->evtchndev, xendev->local_port);

}
