static int net_rx_ok(void *opaque)

{

    struct XenNetDev *netdev = opaque;

    RING_IDX rc, rp;



    if (netdev->xendev.be_state != XenbusStateConnected)

	return 0;



    rc = netdev->rx_ring.req_cons;

    rp = netdev->rx_ring.sring->req_prod;

    xen_rmb();



    if (rc == rp || RING_REQUEST_CONS_OVERFLOW(&netdev->rx_ring, rc)) {

	xen_be_printf(&netdev->xendev, 2, "%s: no rx buffers (%d/%d)\n",

		      __FUNCTION__, rc, rp);

	return 0;

    }

    return 1;

}
