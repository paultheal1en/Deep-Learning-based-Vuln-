void helper_iret_protected(int shift, int next_eip)

{

    int tss_selector, type;

    uint32_t e1, e2;



    /* specific case for TSS */

    if (env->eflags & NT_MASK) {

#ifdef TARGET_X86_64

        if (env->hflags & HF_LMA_MASK)

            raise_exception_err(EXCP0D_GPF, 0);

#endif

        tss_selector = lduw_kernel(env->tr.base + 0);

        if (tss_selector & 4)

            raise_exception_err(EXCP0A_TSS, tss_selector & 0xfffc);

        if (load_segment(&e1, &e2, tss_selector) != 0)

            raise_exception_err(EXCP0A_TSS, tss_selector & 0xfffc);

        type = (e2 >> DESC_TYPE_SHIFT) & 0x17;

        /* NOTE: we check both segment and busy TSS */

        if (type != 3)

            raise_exception_err(EXCP0A_TSS, tss_selector & 0xfffc);

        switch_tss(tss_selector, e1, e2, SWITCH_TSS_IRET, next_eip);

    } else {

        helper_ret_protected(shift, 1, 0);

    }

    env->hflags2 &= ~HF2_NMI_MASK;

#ifdef CONFIG_KQEMU

    if (kqemu_is_ok(env)) {

        CC_OP = CC_OP_EFLAGS;

        env->exception_index = -1;

        cpu_loop_exit();

    }

#endif

}
