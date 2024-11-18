void helper_store_fpscr(CPUPPCState *env, uint64_t arg, uint32_t mask)

{

    /*

     * We use only the 32 LSB of the incoming fpr

     */

    uint32_t prev, new;

    int i;



    prev = env->fpscr;

    new = (uint32_t)arg;

    new &= ~0x60000000;

    new |= prev & 0x60000000;

    for (i = 0; i < 8; i++) {

        if (mask & (1 << i)) {

            env->fpscr &= ~(0xF << (4 * i));

            env->fpscr |= new & (0xF << (4 * i));

        }

    }

    /* Update VX and FEX */

    if (fpscr_ix != 0) {

        env->fpscr |= 1 << FPSCR_VX;

    } else {

        env->fpscr &= ~(1 << FPSCR_VX);

    }

    if ((fpscr_ex & fpscr_eex) != 0) {

        env->fpscr |= 1 << FPSCR_FEX;

        env->exception_index = POWERPC_EXCP_PROGRAM;

        /* XXX: we should compute it properly */

        env->error_code = POWERPC_EXCP_FP;

    } else {

        env->fpscr &= ~(1 << FPSCR_FEX);

    }

    fpscr_set_rounding_mode(env);

}
