static void do_info_profile(Monitor *mon)

{

    int64_t total;

    total = qemu_time;

    if (total == 0)

        total = 1;

    monitor_printf(mon, "async time  %" PRId64 " (%0.3f)\n",

                   dev_time, dev_time / (double)ticks_per_sec);

    monitor_printf(mon, "qemu time   %" PRId64 " (%0.3f)\n",

                   qemu_time, qemu_time / (double)ticks_per_sec);

    monitor_printf(mon, "kqemu time  %" PRId64 " (%0.3f %0.1f%%) count=%"

                        PRId64 " int=%" PRId64 " excp=%" PRId64 " intr=%"

                        PRId64 "\n",

                   kqemu_time, kqemu_time / (double)ticks_per_sec,

                   kqemu_time / (double)total * 100.0,

                   kqemu_exec_count,

                   kqemu_ret_int_count,

                   kqemu_ret_excp_count,

                   kqemu_ret_intr_count);

    qemu_time = 0;

    kqemu_time = 0;

    kqemu_exec_count = 0;

    dev_time = 0;

    kqemu_ret_int_count = 0;

    kqemu_ret_excp_count = 0;

    kqemu_ret_intr_count = 0;

#ifdef CONFIG_KQEMU

    kqemu_record_dump();

#endif

}
