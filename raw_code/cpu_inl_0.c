uint32_t cpu_inl(pio_addr_t addr)

{

    uint32_t val;

    val = ioport_read(2, addr);

    trace_cpu_in(addr, val);

    LOG_IOPORT("inl : %04"FMT_pioaddr" %08"PRIx32"\n", addr, val);

    return val;

}
