void cpu_exec_exit(CPUState *cpu)

{

    if (cpu->cpu_index == -1) {

        /* cpu_index was never allocated by this @cpu or was already freed. */

        return;

    }



    bitmap_clear(cpu_index_map, cpu->cpu_index, 1);

    cpu->cpu_index = -1;

}
