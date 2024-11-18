static ssize_t usbnet_receive(VLANClientState *nc, const uint8_t *buf, size_t size)

{

    USBNetState *s = DO_UPCAST(NICState, nc, nc)->opaque;

    struct rndis_packet_msg_type *msg;



    if (is_rndis(s)) {

        msg = (struct rndis_packet_msg_type *) s->in_buf;

        if (!s->rndis_state == RNDIS_DATA_INITIALIZED)

            return -1;

        if (size + sizeof(struct rndis_packet_msg_type) > sizeof(s->in_buf))

            return -1;



        memset(msg, 0, sizeof(struct rndis_packet_msg_type));

        msg->MessageType = cpu_to_le32(RNDIS_PACKET_MSG);

        msg->MessageLength = cpu_to_le32(size + sizeof(struct rndis_packet_msg_type));

        msg->DataOffset = cpu_to_le32(sizeof(struct rndis_packet_msg_type) - 8);

        msg->DataLength = cpu_to_le32(size);

        /* msg->OOBDataOffset;

         * msg->OOBDataLength;

         * msg->NumOOBDataElements;

         * msg->PerPacketInfoOffset;

         * msg->PerPacketInfoLength;

         * msg->VcHandle;

         * msg->Reserved;

         */

        memcpy(msg + 1, buf, size);

        s->in_len = size + sizeof(struct rndis_packet_msg_type);

    } else {

        if (size > sizeof(s->in_buf))

            return -1;

        memcpy(s->in_buf, buf, size);

        s->in_len = size;

    }

    s->in_ptr = 0;

    return size;

}
