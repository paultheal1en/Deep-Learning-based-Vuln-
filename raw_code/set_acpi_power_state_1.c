static void set_acpi_power_state(IPMIBmcSim *ibs,

                          uint8_t *cmd, unsigned int cmd_len,

                          uint8_t *rsp, unsigned int *rsp_len,

                          unsigned int max_rsp_len)

{

    IPMI_CHECK_CMD_LEN(4);

    ibs->acpi_power_state[0] = cmd[2];

    ibs->acpi_power_state[1] = cmd[3];

}
