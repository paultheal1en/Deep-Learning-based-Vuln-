static int bt_hci_name_req(struct bt_hci_s *hci, bdaddr_t *bdaddr)

{

    struct bt_device_s *slave;

    evt_remote_name_req_complete params;

    int len;



    for (slave = hci->device.net->slave; slave; slave = slave->next)

        if (slave->page_scan && !bacmp(&slave->bd_addr, bdaddr))

            break;

    if (!slave)

        return -ENODEV;



    bt_hci_event_status(hci, HCI_SUCCESS);



    params.status       = HCI_SUCCESS;

    bacpy(&params.bdaddr, &slave->bd_addr);

    len = snprintf(params.name, sizeof(params.name),

                    "%s", slave->lmp_name ?: "");

    memset(params.name + len, 0, sizeof(params.name) - len);

    bt_hci_event(hci, EVT_REMOTE_NAME_REQ_COMPLETE,

                    &params, EVT_REMOTE_NAME_REQ_COMPLETE_SIZE);



    return 0;

}
