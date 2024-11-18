void cpu_outb(CPUState *env, pio_addr_t addr, uint8_t val)

{

    LOG_IOPORT("outb: %04"FMT_pioaddr" %02"PRIx8"\n", addr, val);

    ioport_write(0, addr, val);

#ifdef CONFIG_KQEMU

    if (env)

        env->last_io_time = cpu_get_time_fast();

#endif

}
