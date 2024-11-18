uint16_t cpu_inw(pio_addr_t addr)

{

    uint16_t val;

    val = ioport_read(1, addr);

    trace_cpu_in(addr, val);

    LOG_IOPORT("inw : %04"FMT_pioaddr" %04"PRIx16"\n", addr, val);

    return val;

}
