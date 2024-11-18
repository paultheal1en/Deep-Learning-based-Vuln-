static void bt_submit_sco(struct HCIInfo *info,

                const uint8_t *data, int length)

{

    struct bt_hci_s *hci = hci_from_info(info);

    uint16_t handle;

    int datalen;



    if (length < 3)

        return;



    handle = acl_handle((data[1] << 8) | data[0]);

    datalen = data[2];

    length -= 3;



    if (bt_hci_handle_bad(hci, handle)) {

        fprintf(stderr, "%s: invalid SCO handle %03x\n",

                        __FUNCTION__, handle);

        return;

    }



    if (datalen > length) {

        fprintf(stderr, "%s: SCO packet too short (%iB < %iB)\n",

                        __FUNCTION__, length, datalen);

        return;

    }



    /* TODO */



    /* TODO: increase counter and send EVT_NUM_COMP_PKTS if synchronous

     * Flow Control is enabled.

     * (See Read/Write_Synchronous_Flow_Control_Enable on page 513 and

     * page 514.)  */

}
