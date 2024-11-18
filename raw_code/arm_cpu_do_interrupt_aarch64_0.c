static void arm_cpu_do_interrupt_aarch64(CPUState *cs)

{

    ARMCPU *cpu = ARM_CPU(cs);

    CPUARMState *env = &cpu->env;

    unsigned int new_el = env->exception.target_el;

    target_ulong addr = env->cp15.vbar_el[new_el];

    unsigned int new_mode = aarch64_pstate_mode(new_el, true);



    if (arm_current_el(env) < new_el) {

        if (env->aarch64) {

            addr += 0x400;

        } else {

            addr += 0x600;

        }

    } else if (pstate_read(env) & PSTATE_SP) {

        addr += 0x200;

    }



    switch (cs->exception_index) {

    case EXCP_PREFETCH_ABORT:

    case EXCP_DATA_ABORT:

        env->cp15.far_el[new_el] = env->exception.vaddress;

        qemu_log_mask(CPU_LOG_INT, "...with FAR 0x%" PRIx64 "\n",

                      env->cp15.far_el[new_el]);

        /* fall through */

    case EXCP_BKPT:

    case EXCP_UDEF:

    case EXCP_SWI:

    case EXCP_HVC:

    case EXCP_HYP_TRAP:

    case EXCP_SMC:

        env->cp15.esr_el[new_el] = env->exception.syndrome;

        break;

    case EXCP_IRQ:

    case EXCP_VIRQ:

        addr += 0x80;

        break;

    case EXCP_FIQ:

    case EXCP_VFIQ:

        addr += 0x100;

        break;

    case EXCP_SEMIHOST:

        qemu_log_mask(CPU_LOG_INT,

                      "...handling as semihosting call 0x%" PRIx64 "\n",

                      env->xregs[0]);

        env->xregs[0] = do_arm_semihosting(env);

        return;

    default:

        cpu_abort(cs, "Unhandled exception 0x%x\n", cs->exception_index);

    }



    if (is_a64(env)) {

        env->banked_spsr[aarch64_banked_spsr_index(new_el)] = pstate_read(env);

        aarch64_save_sp(env, arm_current_el(env));

        env->elr_el[new_el] = env->pc;

    } else {

        env->banked_spsr[aarch64_banked_spsr_index(new_el)] = cpsr_read(env);

        if (!env->thumb) {

            env->cp15.esr_el[new_el] |= 1 << 25;

        }

        env->elr_el[new_el] = env->regs[15];



        aarch64_sync_32_to_64(env);



        env->condexec_bits = 0;

    }

    qemu_log_mask(CPU_LOG_INT, "...with ELR 0x%" PRIx64 "\n",

                  env->elr_el[new_el]);



    pstate_write(env, PSTATE_DAIF | new_mode);

    env->aarch64 = 1;

    aarch64_restore_sp(env, new_el);



    env->pc = addr;



    qemu_log_mask(CPU_LOG_INT, "...to EL%d PC 0x%" PRIx64 " PSTATE 0x%x\n",

                  new_el, env->pc, pstate_read(env));

}
