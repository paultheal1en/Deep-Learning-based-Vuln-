void op_mtc0_status (void)

{

    uint32_t val, old;

    uint32_t mask = env->Status_rw_bitmask;



    /* No reverse endianness, no MDMX/DSP, no 64bit ops

       implemented. */

    val = T0 & mask;

    old = env->CP0_Status;

    if (!(val & (1 << CP0St_EXL)) &&

        !(val & (1 << CP0St_ERL)) &&

        !(env->hflags & MIPS_HFLAG_DM) &&

        (val & (1 << CP0St_UM)))

        env->hflags |= MIPS_HFLAG_UM;

#ifdef TARGET_MIPS64

    if ((env->hflags & MIPS_HFLAG_UM) &&

        !(val & (1 << CP0St_PX)) &&

        !(val & (1 << CP0St_UX)))

        env->hflags &= ~MIPS_HFLAG_64;

#endif

    env->CP0_Status = (env->CP0_Status & ~mask) | val;

    if (loglevel & CPU_LOG_EXEC)

        CALL_FROM_TB2(do_mtc0_status_debug, old, val);

    CALL_FROM_TB1(cpu_mips_update_irq, env);

    RETURN();

}
