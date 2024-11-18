static void get_sdr(IPMIBmcSim *ibs,

                    uint8_t *cmd, unsigned int cmd_len,

                    uint8_t *rsp, unsigned int *rsp_len,

                    unsigned int max_rsp_len)

{

    unsigned int pos;

    uint16_t nextrec;

    struct ipmi_sdr_header *sdrh;



    IPMI_CHECK_CMD_LEN(8);

    if (cmd[6]) {

        IPMI_CHECK_RESERVATION(2, ibs->sdr.reservation);

    }

    pos = 0;

    if (sdr_find_entry(&ibs->sdr, cmd[4] | (cmd[5] << 8),

                       &pos, &nextrec)) {

        rsp[2] = IPMI_CC_REQ_ENTRY_NOT_PRESENT;

        return;

    }



    sdrh = (struct ipmi_sdr_header *) &ibs->sdr.sdr[pos];



    if (cmd[6] > ipmi_sdr_length(sdrh)) {

        rsp[2] = IPMI_CC_PARM_OUT_OF_RANGE;

        return;

    }



    IPMI_ADD_RSP_DATA(nextrec & 0xff);

    IPMI_ADD_RSP_DATA((nextrec >> 8) & 0xff);



    if (cmd[7] == 0xff) {

        cmd[7] = ipmi_sdr_length(sdrh) - cmd[6];

    }



    if ((cmd[7] + *rsp_len) > max_rsp_len) {

        rsp[2] = IPMI_CC_CANNOT_RETURN_REQ_NUM_BYTES;

        return;

    }

    memcpy(rsp + *rsp_len, ibs->sdr.sdr + pos + cmd[6], cmd[7]);

    *rsp_len += cmd[7];

}
