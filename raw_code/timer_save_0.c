static void timer_save(QEMUFile *f, void *opaque)

{

    if (cpu_ticks_enabled) {

        hw_error("cannot save state if virtual timers are running");

    }

    qemu_put_be64(f, cpu_ticks_offset);

    qemu_put_be64(f, ticks_per_sec);

    qemu_put_be64(f, cpu_clock_offset);

}
