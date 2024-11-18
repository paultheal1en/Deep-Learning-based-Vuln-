static void qemu_tcg_wait_io_event(CPUState *cpu)

{

    while (all_cpu_threads_idle()) {

        stop_tcg_kick_timer();

        qemu_cond_wait(cpu->halt_cond, &qemu_global_mutex);

    }



    start_tcg_kick_timer();



    CPU_FOREACH(cpu) {

        qemu_wait_io_event_common(cpu);

    }

}
