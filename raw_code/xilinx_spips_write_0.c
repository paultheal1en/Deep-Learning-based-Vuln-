static void xilinx_spips_write(void *opaque, hwaddr addr,

                                        uint64_t value, unsigned size)

{

    int mask = ~0;

    int man_start_com = 0;

    XilinxSPIPS *s = opaque;



    DB_PRINT("addr=" TARGET_FMT_plx " = %x\n", addr, (unsigned)value);

    addr >>= 2;

    switch (addr) {

    case R_CONFIG:

        mask = 0x0002FFFF;

        if (value & MAN_START_COM) {

            man_start_com = 1;

        }

        break;

    case R_INTR_STATUS:

        mask = IXR_ALL;

        s->regs[R_INTR_STATUS] &= ~(mask & value);

        goto no_reg_update;

    case R_INTR_DIS:

        mask = IXR_ALL;

        s->regs[R_INTR_MASK] &= ~(mask & value);

        goto no_reg_update;

    case R_INTR_EN:

        mask = IXR_ALL;

        s->regs[R_INTR_MASK] |= mask & value;

        goto no_reg_update;

    case R_EN:

        mask = 0x1;

        break;

    case R_SLAVE_IDLE_COUNT:

        mask = 0xFF;

        break;

    case R_RX_DATA:

    case R_INTR_MASK:

    case R_MOD_ID:

        mask = 0;

        break;

    case R_TX_DATA:

        tx_data_bytes(s, (uint32_t)value, s->num_txrx_bytes);

        goto no_reg_update;

    case R_TXD1:

        tx_data_bytes(s, (uint32_t)value, 1);

        goto no_reg_update;

    case R_TXD2:

        tx_data_bytes(s, (uint32_t)value, 2);

        goto no_reg_update;

    case R_TXD3:

        tx_data_bytes(s, (uint32_t)value, 3);

        goto no_reg_update;

    }

    s->regs[addr] = (s->regs[addr] & ~mask) | (value & mask);

no_reg_update:

    xilinx_spips_update_cs_lines(s);

    if ((man_start_com && s->regs[R_CONFIG] & MAN_START_EN) ||

            (fifo8_is_empty(&s->tx_fifo) && s->regs[R_CONFIG] & MAN_START_EN)) {

        xilinx_spips_flush_txfifo(s);

    }

    xilinx_spips_update_cs_lines(s);

    xilinx_spips_update_ixr(s);

}
