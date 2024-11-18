static void do_info_kqemu(Monitor *mon)

{

#ifdef CONFIG_KQEMU

    CPUState *env;

    int val;

    val = 0;

    env = mon_get_cpu();

    if (!env) {

        monitor_printf(mon, "No cpu initialized yet");

        return;

    }

    val = env->kqemu_enabled;

    monitor_printf(mon, "kqemu support: ");

    switch(val) {

    default:

    case 0:

        monitor_printf(mon, "disabled\n");

        break;

    case 1:

        monitor_printf(mon, "enabled for user code\n");

        break;

    case 2:

        monitor_printf(mon, "enabled for user and kernel code\n");

        break;

    }

#else

    monitor_printf(mon, "kqemu support: not compiled\n");

#endif

}
