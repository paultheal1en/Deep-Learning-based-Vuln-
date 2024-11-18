static void __cpu_ppc_store_decr(PowerPCCPU *cpu, uint64_t *nextp,

                                 QEMUTimer *timer,

                                 void (*raise_excp)(PowerPCCPU *),

                                 uint32_t decr, uint32_t value,

                                 int is_excp)

{

    CPUPPCState *env = &cpu->env;

    ppc_tb_t *tb_env = env->tb_env;

    uint64_t now, next;



    LOG_TB("%s: %08" PRIx32 " => %08" PRIx32 "\n", __func__,

                decr, value);



    if (kvm_enabled()) {

        /* KVM handles decrementer exceptions, we don't need our own timer */

        return;

    }



    now = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);

    next = now + muldiv64(value, get_ticks_per_sec(), tb_env->decr_freq);

    if (is_excp) {

        next += *nextp - now;

    }

    if (next == now) {

        next++;

    }

    *nextp = next;

    /* Adjust timer */

    timer_mod(timer, next);



    /* If we set a negative value and the decrementer was positive, raise an

     * exception.

     */

    if ((tb_env->flags & PPC_DECR_UNDERFLOW_TRIGGERED)

        && (value & 0x80000000)

        && !(decr & 0x80000000)) {

        (*raise_excp)(cpu);

    }

}
