static void qemu_wait_io_event_common(CPUState *cpu)

{

    if (cpu->stop) {

        cpu->stop = false;

        cpu->stopped = true;

        qemu_cond_broadcast(&qemu_pause_cond);

    }

    process_queued_cpu_work(cpu);

    cpu->thread_kicked = false;

}
