static void add_sel_entry(IPMIBmcSim *ibs,

                          uint8_t *cmd, unsigned int cmd_len,

                          uint8_t *rsp, unsigned int *rsp_len,

                          unsigned int max_rsp_len)

{

    IPMI_CHECK_CMD_LEN(18);

    if (sel_add_event(ibs, cmd + 2)) {

        rsp[2] = IPMI_CC_OUT_OF_SPACE;

        return;

    }

    /* sel_add_event fills in the record number. */

    IPMI_ADD_RSP_DATA(cmd[2]);

    IPMI_ADD_RSP_DATA(cmd[3]);

}
