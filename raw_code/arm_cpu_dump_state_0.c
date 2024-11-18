void arm_cpu_dump_state(CPUState *cs, FILE *f, fprintf_function cpu_fprintf,

                        int flags)

{

    ARMCPU *cpu = ARM_CPU(cs);

    CPUARMState *env = &cpu->env;

    int i;



    if (is_a64(env)) {

        aarch64_cpu_dump_state(cs, f, cpu_fprintf, flags);

        return;

    }



    for(i=0;i<16;i++) {

        cpu_fprintf(f, "R%02d=%08x", i, env->regs[i]);

        if ((i % 4) == 3)

            cpu_fprintf(f, "\n");

        else

            cpu_fprintf(f, " ");

    }



    if (arm_feature(env, ARM_FEATURE_M)) {

        uint32_t xpsr = xpsr_read(env);

        const char *mode;



        if (xpsr & XPSR_EXCP) {

            mode = "handler";

        } else {

            if (env->v7m.control & R_V7M_CONTROL_NPRIV_MASK) {

                mode = "unpriv-thread";

            } else {

                mode = "priv-thread";

            }

        }



        cpu_fprintf(f, "XPSR=%08x %c%c%c%c %c %s\n",

                    xpsr,

                    xpsr & XPSR_N ? 'N' : '-',

                    xpsr & XPSR_Z ? 'Z' : '-',

                    xpsr & XPSR_C ? 'C' : '-',

                    xpsr & XPSR_V ? 'V' : '-',

                    xpsr & XPSR_T ? 'T' : 'A',

                    mode);

    } else {

        uint32_t psr = cpsr_read(env);

        const char *ns_status = "";



        if (arm_feature(env, ARM_FEATURE_EL3) &&

            (psr & CPSR_M) != ARM_CPU_MODE_MON) {

            ns_status = env->cp15.scr_el3 & SCR_NS ? "NS " : "S ";

        }



        cpu_fprintf(f, "PSR=%08x %c%c%c%c %c %s%s%d\n",

                    psr,

                    psr & CPSR_N ? 'N' : '-',

                    psr & CPSR_Z ? 'Z' : '-',

                    psr & CPSR_C ? 'C' : '-',

                    psr & CPSR_V ? 'V' : '-',

                    psr & CPSR_T ? 'T' : 'A',

                    ns_status,

                    cpu_mode_names[psr & 0xf], (psr & 0x10) ? 32 : 26);

    }



    if (flags & CPU_DUMP_FPU) {

        int numvfpregs = 0;

        if (arm_feature(env, ARM_FEATURE_VFP)) {

            numvfpregs += 16;

        }

        if (arm_feature(env, ARM_FEATURE_VFP3)) {

            numvfpregs += 16;

        }

        for (i = 0; i < numvfpregs; i++) {

            uint64_t v = float64_val(env->vfp.regs[i]);

            cpu_fprintf(f, "s%02d=%08x s%02d=%08x d%02d=%016" PRIx64 "\n",

                        i * 2, (uint32_t)v,

                        i * 2 + 1, (uint32_t)(v >> 32),

                        i, v);

        }

        cpu_fprintf(f, "FPSCR: %08x\n", (int)env->vfp.xregs[ARM_VFP_FPSCR]);

    }

}
