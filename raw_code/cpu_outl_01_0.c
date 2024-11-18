void cpu_outl(CPUState *env, pio_addr_t addr, uint32_t val)

{

    LOG_IOPORT("outl: %04"FMT_pioaddr" %08"PRIx32"\n", addr, val);

    ioport_write(2, addr, val);

#ifdef CONFIG_KQEMU

    if (env)

        env->last_io_time = cpu_get_time_fast();

#endif

}
