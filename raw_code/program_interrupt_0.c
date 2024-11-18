void program_interrupt(CPUS390XState *env, uint32_t code, int ilen)

{

    S390CPU *cpu = s390_env_get_cpu(env);



    qemu_log_mask(CPU_LOG_INT, "program interrupt at %#" PRIx64 "\n",

                  env->psw.addr);



    if (kvm_enabled()) {

        kvm_s390_program_interrupt(cpu, code);

    } else if (tcg_enabled()) {

        tcg_s390_program_interrupt(env, code, ilen);

    } else {

        g_assert_not_reached();

    }

}
