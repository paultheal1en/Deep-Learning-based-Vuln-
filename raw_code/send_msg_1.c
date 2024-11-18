static void send_msg(IPMIBmcSim *ibs,

                     uint8_t *cmd, unsigned int cmd_len,

                     uint8_t *rsp, unsigned int *rsp_len,

                     unsigned int max_rsp_len)

{

    IPMIInterface *s = ibs->parent.intf;

    IPMIInterfaceClass *k = IPMI_INTERFACE_GET_CLASS(s);

    IPMIRcvBufEntry *msg;

    uint8_t *buf;

    uint8_t netfn, rqLun, rsLun, rqSeq;



    IPMI_CHECK_CMD_LEN(3);



    if (cmd[2] != 0) {

        /* We only handle channel 0 with no options */

        rsp[2] = IPMI_CC_INVALID_DATA_FIELD;

        return;

    }



    IPMI_CHECK_CMD_LEN(10);

    if (cmd[3] != 0x40) {

        /* We only emulate a MC at address 0x40. */

        rsp[2] = 0x83; /* NAK on write */

        return;

    }



    cmd += 3; /* Skip the header. */

    cmd_len -= 3;



    /*

     * At this point we "send" the message successfully.  Any error will

     * be returned in the response.

     */

    if (ipmb_checksum(cmd, cmd_len, 0) != 0 ||

        cmd[3] != 0x20) { /* Improper response address */

        return; /* No response */

    }



    netfn = cmd[1] >> 2;

    rqLun = cmd[4] & 0x3;

    rsLun = cmd[1] & 0x3;

    rqSeq = cmd[4] >> 2;



    if (rqLun != 2) {

        /* We only support LUN 2 coming back to us. */

        return;

    }



    msg = g_malloc(sizeof(*msg));

    msg->buf[0] = ((netfn | 1) << 2) | rqLun; /* NetFN, and make a response */

    msg->buf[1] = ipmb_checksum(msg->buf, 1, 0);

    msg->buf[2] = cmd[0]; /* rsSA */

    msg->buf[3] = (rqSeq << 2) | rsLun;

    msg->buf[4] = cmd[5]; /* Cmd */

    msg->buf[5] = 0; /* Completion Code */

    msg->len = 6;



    if ((cmd[1] >> 2) != IPMI_NETFN_APP || cmd[5] != IPMI_CMD_GET_DEVICE_ID) {

        /* Not a command we handle. */

        msg->buf[5] = IPMI_CC_INVALID_CMD;

        goto end_msg;

    }



    buf = msg->buf + msg->len; /* After the CC */

    buf[0] = 0;

    buf[1] = 0;

    buf[2] = 0;

    buf[3] = 0;

    buf[4] = 0x51;

    buf[5] = 0;

    buf[6] = 0;

    buf[7] = 0;

    buf[8] = 0;

    buf[9] = 0;

    buf[10] = 0;

    msg->len += 11;



 end_msg:

    msg->buf[msg->len] = ipmb_checksum(msg->buf, msg->len, 0);

    msg->len++;

    qemu_mutex_lock(&ibs->lock);

    QTAILQ_INSERT_TAIL(&ibs->rcvbufs, msg, entry);

    ibs->msg_flags |= IPMI_BMC_MSG_FLAG_RCV_MSG_QUEUE;

    k->set_atn(s, 1, attn_irq_enabled(ibs));

    qemu_mutex_unlock(&ibs->lock);

}
