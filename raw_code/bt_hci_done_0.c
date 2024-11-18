static void bt_hci_done(struct HCIInfo *info)

{

    struct bt_hci_s *hci = hci_from_info(info);

    int handle;



    bt_device_done(&hci->device);



    if (hci->device.lmp_name)

        g_free((void *) hci->device.lmp_name);



    /* Be gentle and send DISCONNECT to all connected peers and those

     * currently waiting for us to accept or reject a connection request.

     * This frees the links.  */

    if (hci->conn_req_host) {

        bt_hci_connection_reject(hci,

                                 hci->conn_req_host, HCI_OE_POWER_OFF);

        return;

    }



    for (handle = HCI_HANDLE_OFFSET;

                    handle < (HCI_HANDLE_OFFSET | HCI_HANDLES_MAX); handle ++)

        if (!bt_hci_handle_bad(hci, handle))

            bt_hci_disconnect(hci, handle, HCI_OE_POWER_OFF);



    /* TODO: this is not enough actually, there may be slaves from whom

     * we have requested a connection who will soon (or not) respond with

     * an accept or a reject, so we should also check if hci->lm.connecting

     * is non-zero and if so, avoid freeing the hci but otherwise disappear

     * from all qemu social life (e.g. stop scanning and request to be

     * removed from s->device.net) and arrange for

     * s->device.lmp_connection_complete to free the remaining bits once

     * hci->lm.awaiting_bdaddr[] is empty.  */



    timer_free(hci->lm.inquiry_done);

    timer_free(hci->lm.inquiry_next);

    timer_free(hci->conn_accept_timer);



    g_free(hci);

}
