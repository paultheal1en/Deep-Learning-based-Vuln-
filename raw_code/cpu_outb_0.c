void cpu_outb(pio_addr_t addr, uint8_t val)

{

    LOG_IOPORT("outb: %04"FMT_pioaddr" %02"PRIx8"\n", addr, val);

    trace_cpu_out(addr, val);

    ioport_write(0, addr, val);

}
