static void xlnx_zynqmp_qspips_reset(DeviceState *d)

{

    XlnxZynqMPQSPIPS *s = XLNX_ZYNQMP_QSPIPS(d);

    int i;



    xilinx_spips_reset(d);



    for (i = 0; i < XLNX_ZYNQMP_SPIPS_R_MAX; i++) {

        s->regs[i] = 0;

    }

    fifo8_reset(&s->rx_fifo_g);

    fifo8_reset(&s->rx_fifo_g);

    fifo32_reset(&s->fifo_g);

    s->regs[R_INTR_STATUS] = R_INTR_STATUS_RESET;

    s->regs[R_GPIO] = 1;

    s->regs[R_LPBK_DLY_ADJ] = R_LPBK_DLY_ADJ_RESET;

    s->regs[R_GQSPI_GFIFO_THRESH] = 0x10;

    s->regs[R_MOD_ID] = 0x01090101;

    s->regs[R_GQSPI_IMR] = R_GQSPI_IMR_RESET;

    s->regs[R_GQSPI_TX_THRESH] = 1;

    s->regs[R_GQSPI_RX_THRESH] = 1;

    s->regs[R_GQSPI_GPIO] = 1;

    s->regs[R_GQSPI_LPBK_DLY_ADJ] = R_GQSPI_LPBK_DLY_ADJ_RESET;

    s->regs[R_GQSPI_MOD_ID] = R_GQSPI_MOD_ID_RESET;

    s->regs[R_QSPIDMA_DST_CTRL] = R_QSPIDMA_DST_CTRL_RESET;

    s->regs[R_QSPIDMA_DST_I_MASK] = R_QSPIDMA_DST_I_MASK_RESET;

    s->regs[R_QSPIDMA_DST_CTRL2] = R_QSPIDMA_DST_CTRL2_RESET;

    s->man_start_com_g = false;

    s->gqspi_irqline = 0;

    xlnx_zynqmp_qspips_update_ixr(s);

}
