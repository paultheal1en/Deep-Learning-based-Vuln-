bool qemu_run_timers(QEMUClock *clock)

{

    return qemu_clock_run_timers(clock->type);

}
