static void bt_submit_acl(struct HCIInfo *info,

                const uint8_t *data, int length)

{

    struct bt_hci_s *hci = hci_from_info(info);

    uint16_t handle;

    int datalen, flags;

    struct bt_link_s *link;



    if (length < HCI_ACL_HDR_SIZE) {

        fprintf(stderr, "%s: ACL packet too short (%iB)\n",

                        __FUNCTION__, length);

        return;

    }



    handle = acl_handle((data[1] << 8) | data[0]);

    flags = acl_flags((data[1] << 8) | data[0]);

    datalen = (data[3] << 8) | data[2];

    data += HCI_ACL_HDR_SIZE;

    length -= HCI_ACL_HDR_SIZE;



    if (bt_hci_handle_bad(hci, handle)) {

        fprintf(stderr, "%s: invalid ACL handle %03x\n",

                        __FUNCTION__, handle);

        /* TODO: signal an error */

        return;

    }

    handle &= ~HCI_HANDLE_OFFSET;



    if (datalen > length) {

        fprintf(stderr, "%s: ACL packet too short (%iB < %iB)\n",

                        __FUNCTION__, length, datalen);

        return;

    }



    link = hci->lm.handle[handle].link;



    if ((flags & ~3) == ACL_ACTIVE_BCAST) {

        if (!hci->asb_handle)

            hci->asb_handle = handle;

        else if (handle != hci->asb_handle) {

            fprintf(stderr, "%s: Bad handle %03x in Active Slave Broadcast\n",

                            __FUNCTION__, handle);

            /* TODO: signal an error */

            return;

        }



        /* TODO */

    }



    if ((flags & ~3) == ACL_PICO_BCAST) {

        if (!hci->psb_handle)

            hci->psb_handle = handle;

        else if (handle != hci->psb_handle) {

            fprintf(stderr, "%s: Bad handle %03x in Parked Slave Broadcast\n",

                            __FUNCTION__, handle);

            /* TODO: signal an error */

            return;

        }



        /* TODO */

    }



    /* TODO: increase counter and send EVT_NUM_COMP_PKTS */

    bt_hci_event_num_comp_pkts(hci, handle | HCI_HANDLE_OFFSET, 1);



    /* Do this last as it can trigger further events even in this HCI */

    hci->lm.handle[handle].lmp_acl_data(link, data,

                    (flags & 3) == ACL_START, length);

}
