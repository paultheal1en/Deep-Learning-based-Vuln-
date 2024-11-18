static void set_sel_time(IPMIBmcSim *ibs,

                         uint8_t *cmd, unsigned int cmd_len,

                         uint8_t *rsp, unsigned int *rsp_len,

                         unsigned int max_rsp_len)

{

    uint32_t val;

    struct ipmi_time now;



    IPMI_CHECK_CMD_LEN(6);

    val = cmd[2] | (cmd[3] << 8) | (cmd[4] << 16) | (cmd[5] << 24);

    ipmi_gettime(&now);

    ibs->sel.time_offset = now.tv_sec - ((long) val);

}
