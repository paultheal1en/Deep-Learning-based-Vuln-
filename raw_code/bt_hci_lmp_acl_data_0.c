static inline void bt_hci_lmp_acl_data(struct bt_hci_s *hci, uint16_t handle,

                const uint8_t *data, int start, int len)

{

    struct hci_acl_hdr *pkt = (void *) hci->acl_buf;



    /* TODO: packet flags */

    /* TODO: avoid memcpy'ing */



    if (len + HCI_ACL_HDR_SIZE > sizeof(hci->acl_buf)) {

        fprintf(stderr, "%s: can't take ACL packets %i bytes long\n",

                        __FUNCTION__, len);

        return;

    }

    memcpy(hci->acl_buf + HCI_ACL_HDR_SIZE, data, len);



    pkt->handle = cpu_to_le16(

                    acl_handle_pack(handle, start ? ACL_START : ACL_CONT));

    pkt->dlen = cpu_to_le16(len);

    hci->info.acl_recv(hci->info.opaque,

                    hci->acl_buf, len + HCI_ACL_HDR_SIZE);

}
