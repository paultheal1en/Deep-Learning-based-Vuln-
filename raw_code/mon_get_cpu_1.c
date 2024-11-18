CPUState *mon_get_cpu(void)

{

    if (!cur_mon->mon_cpu) {

        monitor_set_cpu(0);

    }

    cpu_synchronize_state(cur_mon->mon_cpu);

    return cur_mon->mon_cpu;

}
