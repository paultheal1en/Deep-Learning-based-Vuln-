static void do_interrupt_real(int intno, int is_int, int error_code,

                                 unsigned int next_eip)

{

    SegmentCache *dt;

    uint8_t *ptr, *ssp;

    int selector;

    uint32_t offset, esp;

    uint32_t old_cs, old_eip;



    /* real mode (simpler !) */

    dt = &env->idt;

    if (intno * 4 + 3 > dt->limit)

        raise_exception_err(EXCP0D_GPF, intno * 8 + 2);

    ptr = dt->base + intno * 4;

    offset = lduw(ptr);

    selector = lduw(ptr + 2);

    esp = env->regs[R_ESP];

    ssp = env->segs[R_SS].base;

    if (is_int)

        old_eip = next_eip;

    else

        old_eip = env->eip;

    old_cs = env->segs[R_CS].selector;

    esp -= 2;

    stw(ssp + (esp & 0xffff), compute_eflags());

    esp -= 2;

    stw(ssp + (esp & 0xffff), old_cs);

    esp -= 2;

    stw(ssp + (esp & 0xffff), old_eip);

    

    /* update processor state */

    env->regs[R_ESP] = (env->regs[R_ESP] & ~0xffff) | (esp & 0xffff);

    env->eip = offset;

    env->segs[R_CS].selector = selector;

    env->segs[R_CS].base = (uint8_t *)(selector << 4);

    env->eflags &= ~(IF_MASK | TF_MASK | AC_MASK | RF_MASK);

}
