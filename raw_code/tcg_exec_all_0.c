static void tcg_exec_all(void)

{

    int r;



    /* Account partial waits to QEMU_CLOCK_VIRTUAL.  */

    qemu_clock_warp(QEMU_CLOCK_VIRTUAL);



    if (next_cpu == NULL) {

        next_cpu = first_cpu;

    }

    for (; next_cpu != NULL && !exit_request; next_cpu = CPU_NEXT(next_cpu)) {

        CPUState *cpu = next_cpu;



        qemu_clock_enable(QEMU_CLOCK_VIRTUAL,

                          (cpu->singlestep_enabled & SSTEP_NOTIMER) == 0);



        if (cpu_can_run(cpu)) {

            r = tcg_cpu_exec(cpu);

            if (r == EXCP_DEBUG) {

                cpu_handle_guest_debug(cpu);

                break;

            }

        } else if (cpu->stop || cpu->stopped) {

            break;

        }

    }



    /* Pairs with smp_wmb in qemu_cpu_kick.  */

    atomic_mb_set(&exit_request, 0);

}
