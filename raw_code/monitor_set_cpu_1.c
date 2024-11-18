int monitor_set_cpu(int cpu_index)

{

    CPUState *cpu;



    cpu = qemu_get_cpu(cpu_index);

    if (cpu == NULL) {

        return -1;

    }

    cur_mon->mon_cpu = cpu;

    return 0;

}
