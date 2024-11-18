static void xilinx_spips_reset(DeviceState *d)

{

    XilinxSPIPS *s = XILINX_SPIPS(d);



    int i;

    for (i = 0; i < XLNX_SPIPS_R_MAX; i++) {

        s->regs[i] = 0;

    }



    fifo8_reset(&s->rx_fifo);

    fifo8_reset(&s->rx_fifo);

    /* non zero resets */

    s->regs[R_CONFIG] |= MODEFAIL_GEN_EN;

    s->regs[R_SLAVE_IDLE_COUNT] = 0xFF;

    s->regs[R_TX_THRES] = 1;

    s->regs[R_RX_THRES] = 1;

    /* FIXME: move magic number definition somewhere sensible */

    s->regs[R_MOD_ID] = 0x01090106;

    s->regs[R_LQSPI_CFG] = R_LQSPI_CFG_RESET;

    s->link_state = 1;

    s->link_state_next = 1;

    s->link_state_next_when = 0;

    s->snoop_state = SNOOP_CHECKING;

    s->cmd_dummies = 0;

    s->man_start_com = false;

    xilinx_spips_update_ixr(s);

    xilinx_spips_update_cs_lines(s);

}
