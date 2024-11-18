void helper_iret_real(int shift)

{

    uint32_t sp, new_cs, new_eip, new_eflags, new_esp;

    uint8_t *ssp;

    int eflags_mask;

    

    sp = env->regs[R_ESP] & 0xffff;

    ssp = env->segs[R_SS].base + sp;

    if (shift == 1) {

        /* 32 bits */

        new_eflags = ldl(ssp + 8);

        new_cs = ldl(ssp + 4) & 0xffff;

        new_eip = ldl(ssp) & 0xffff;

    } else {

        /* 16 bits */

        new_eflags = lduw(ssp + 4);

        new_cs = lduw(ssp + 2);

        new_eip = lduw(ssp);

    }

    new_esp = sp + (6 << shift);

    env->regs[R_ESP] = (env->regs[R_ESP] & 0xffff0000) | 

        (new_esp & 0xffff);

    load_seg_vm(R_CS, new_cs);

    env->eip = new_eip;

    eflags_mask = FL_UPDATE_CPL0_MASK;

    if (shift == 0)

        eflags_mask &= 0xffff;

    load_eflags(new_eflags, eflags_mask);

}
