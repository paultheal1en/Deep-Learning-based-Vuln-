uint32_t cpu_inl(CPUState *env, pio_addr_t addr)

{

    uint32_t val;

    val = ioport_read(2, addr);

    LOG_IOPORT("inl : %04"FMT_pioaddr" %08"PRIx32"\n", addr, val);

#ifdef CONFIG_KQEMU

    if (env)

        env->last_io_time = cpu_get_time_fast();

#endif

    return val;

}
