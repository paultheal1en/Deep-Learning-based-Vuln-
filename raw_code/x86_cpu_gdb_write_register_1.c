int x86_cpu_gdb_write_register(CPUState *cs, uint8_t *mem_buf, int n)

{

    X86CPU *cpu = X86_CPU(cs);

    CPUX86State *env = &cpu->env;

    uint32_t tmp;



    if (n < CPU_NB_REGS) {

        if (TARGET_LONG_BITS == 64 && env->hflags & HF_CS64_MASK) {

            env->regs[gpr_map[n]] = ldtul_p(mem_buf);

            return sizeof(target_ulong);

        } else if (n < CPU_NB_REGS32) {

            n = gpr_map32[n];

            env->regs[n] &= ~0xffffffffUL;

            env->regs[n] |= (uint32_t)ldl_p(mem_buf);

            return 4;

        }

    } else if (n >= IDX_FP_REGS && n < IDX_FP_REGS + 8) {

#ifdef USE_X86LDOUBLE

        /* FIXME: byteswap float values - after fixing fpregs layout. */

        memcpy(&env->fpregs[n - IDX_FP_REGS], mem_buf, 10);

#endif

        return 10;

    } else if (n >= IDX_XMM_REGS && n < IDX_XMM_REGS + CPU_NB_REGS) {

        n -= IDX_XMM_REGS;

        if (n < CPU_NB_REGS32 ||

            (TARGET_LONG_BITS == 64 && env->hflags & HF_CS64_MASK)) {

            env->xmm_regs[n].XMM_Q(0) = ldq_p(mem_buf);

            env->xmm_regs[n].XMM_Q(1) = ldq_p(mem_buf + 8);

            return 16;

        }

    } else {

        switch (n) {

        case IDX_IP_REG:

            if (TARGET_LONG_BITS == 64 && env->hflags & HF_CS64_MASK) {

                env->eip = ldq_p(mem_buf);

                return 8;

            } else {

                env->eip &= ~0xffffffffUL;

                env->eip |= (uint32_t)ldl_p(mem_buf);

                return 4;

            }

        case IDX_FLAGS_REG:

            env->eflags = ldl_p(mem_buf);

            return 4;



        case IDX_SEG_REGS:

            return x86_cpu_gdb_load_seg(cpu, R_CS, mem_buf);

        case IDX_SEG_REGS + 1:

            return x86_cpu_gdb_load_seg(cpu, R_SS, mem_buf);

        case IDX_SEG_REGS + 2:

            return x86_cpu_gdb_load_seg(cpu, R_DS, mem_buf);

        case IDX_SEG_REGS + 3:

            return x86_cpu_gdb_load_seg(cpu, R_ES, mem_buf);

        case IDX_SEG_REGS + 4:

            return x86_cpu_gdb_load_seg(cpu, R_FS, mem_buf);

        case IDX_SEG_REGS + 5:

            return x86_cpu_gdb_load_seg(cpu, R_GS, mem_buf);



        case IDX_FP_REGS + 8:

            env->fpuc = ldl_p(mem_buf);

            return 4;

        case IDX_FP_REGS + 9:

            tmp = ldl_p(mem_buf);

            env->fpstt = (tmp >> 11) & 7;

            env->fpus = tmp & ~0x3800;

            return 4;

        case IDX_FP_REGS + 10: /* ftag */

            return 4;

        case IDX_FP_REGS + 11: /* fiseg */

            return 4;

        case IDX_FP_REGS + 12: /* fioff */

            return 4;

        case IDX_FP_REGS + 13: /* foseg */

            return 4;

        case IDX_FP_REGS + 14: /* fooff */

            return 4;

        case IDX_FP_REGS + 15: /* fop */

            return 4;



        case IDX_MXCSR_REG:

            env->mxcsr = ldl_p(mem_buf);

            return 4;

        }

    }

    /* Unrecognised register.  */

    return 0;

}
