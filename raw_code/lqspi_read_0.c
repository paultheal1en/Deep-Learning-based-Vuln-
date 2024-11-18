lqspi_read(void *opaque, hwaddr addr, unsigned int size)

{

    int i;

    XilinxQSPIPS *q = opaque;

    XilinxSPIPS *s = opaque;

    uint32_t ret;



    if (addr >= q->lqspi_cached_addr &&

            addr <= q->lqspi_cached_addr + LQSPI_CACHE_SIZE - 4) {

        ret = q->lqspi_buf[(addr - q->lqspi_cached_addr) >> 2];

        DB_PRINT("addr: %08x, data: %08x\n", (unsigned)addr, (unsigned)ret);

        return ret;

    } else {

        int flash_addr = (addr / num_effective_busses(s));

        int slave = flash_addr >> LQSPI_ADDRESS_BITS;

        int cache_entry = 0;



        DB_PRINT("config reg status: %08x\n", s->regs[R_LQSPI_CFG]);



        fifo8_reset(&s->tx_fifo);

        fifo8_reset(&s->rx_fifo);



        s->regs[R_CONFIG] &= ~CS;

        s->regs[R_CONFIG] |= ((~(1 << slave) << CS_SHIFT) & CS) | MANUAL_CS;

        xilinx_spips_update_cs_lines(s);



        /* instruction */

        DB_PRINT("pushing read instruction: %02x\n",

                 (uint8_t)(s->regs[R_LQSPI_CFG] & LQSPI_CFG_INST_CODE));

        fifo8_push(&s->tx_fifo, s->regs[R_LQSPI_CFG] & LQSPI_CFG_INST_CODE);

        /* read address */

        DB_PRINT("pushing read address %06x\n", flash_addr);

        fifo8_push(&s->tx_fifo, (uint8_t)(flash_addr >> 16));

        fifo8_push(&s->tx_fifo, (uint8_t)(flash_addr >> 8));

        fifo8_push(&s->tx_fifo, (uint8_t)flash_addr);

        /* mode bits */

        if (s->regs[R_LQSPI_CFG] & LQSPI_CFG_MODE_EN) {

            fifo8_push(&s->tx_fifo, extract32(s->regs[R_LQSPI_CFG],

                                              LQSPI_CFG_MODE_SHIFT,

                                              LQSPI_CFG_MODE_WIDTH));

        }

        /* dummy bytes */

        for (i = 0; i < (extract32(s->regs[R_LQSPI_CFG], LQSPI_CFG_DUMMY_SHIFT,

                                   LQSPI_CFG_DUMMY_WIDTH)); ++i) {

            DB_PRINT("pushing dummy byte\n");

            fifo8_push(&s->tx_fifo, 0);

        }

        xilinx_spips_update_cs_lines(s);

        xilinx_spips_flush_txfifo(s);

        fifo8_reset(&s->rx_fifo);



        DB_PRINT("starting QSPI data read\n");



        for (i = 0; i < LQSPI_CACHE_SIZE / 4; ++i) {

            tx_data_bytes(s, 0, 4);

            xilinx_spips_flush_txfifo(s);

            rx_data_bytes(s, &q->lqspi_buf[cache_entry], 4);

            cache_entry++;

        }

        xilinx_spips_update_cs_lines(s);



        s->regs[R_CONFIG] |= CS;

        xilinx_spips_update_cs_lines(s);



        q->lqspi_cached_addr = addr;

        return lqspi_read(opaque, addr, size);

    }

}
