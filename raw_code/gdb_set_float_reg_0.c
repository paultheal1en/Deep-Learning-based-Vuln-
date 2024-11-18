static int gdb_set_float_reg(CPUPPCState *env, uint8_t *mem_buf, int n)

{

    if (n < 32) {

        env->fpr[n] = ldfq_p(mem_buf);

        return 8;

    }

    if (n == 32) {

        /* FPSCR not implemented  */

        return 4;

    }

    return 0;

}
