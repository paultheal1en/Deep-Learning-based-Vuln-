void qemu_cpu_kick(CPUState *cpu)

{

    qemu_cond_broadcast(cpu->halt_cond);

    if (tcg_enabled()) {

        cpu_exit(cpu);

        /* Also ensure current RR cpu is kicked */

        qemu_cpu_kick_rr_cpu();

    } else {

        if (hax_enabled()) {

            /*

             * FIXME: race condition with the exit_request check in

             * hax_vcpu_hax_exec

             */

            cpu->exit_request = 1;

        }

        qemu_cpu_kick_thread(cpu);

    }

}
