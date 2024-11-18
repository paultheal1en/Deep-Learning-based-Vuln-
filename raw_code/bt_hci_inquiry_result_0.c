static void bt_hci_inquiry_result(struct bt_hci_s *hci,

                struct bt_device_s *slave)

{

    if (!slave->inquiry_scan || !hci->lm.responses_left)

        return;



    hci->lm.responses_left --;

    hci->lm.responses ++;



    switch (hci->lm.inquiry_mode) {

    case 0x00:

        bt_hci_inquiry_result_standard(hci, slave);

        return;

    case 0x01:

        bt_hci_inquiry_result_with_rssi(hci, slave);

        return;

    default:

        fprintf(stderr, "%s: bad inquiry mode %02x\n", __FUNCTION__,

                        hci->lm.inquiry_mode);

        exit(-1);

    }

}
