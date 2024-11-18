void op_mtc0_ebase (void)

{

    /* vectored interrupts not implemented */

    /* Multi-CPU not implemented */

    env->CP0_EBase = (int32_t)0x80000000 | (T0 & 0x3FFFF000);

    RETURN();

}
