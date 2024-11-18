static void clr_msg_flags(IPMIBmcSim *ibs,

                          uint8_t *cmd, unsigned int cmd_len,

                          uint8_t *rsp, unsigned int *rsp_len,

                          unsigned int max_rsp_len)

{

    IPMIInterface *s = ibs->parent.intf;

    IPMIInterfaceClass *k = IPMI_INTERFACE_GET_CLASS(s);



    IPMI_CHECK_CMD_LEN(3);

    ibs->msg_flags &= ~cmd[2];

    k->set_atn(s, attn_set(ibs), attn_irq_enabled(ibs));

}
