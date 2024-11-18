static void virt_acpi_get_cpu_info(VirtAcpiCpuInfo *cpuinfo)

{

    CPUState *cpu;



    memset(cpuinfo->found_cpus, 0, sizeof cpuinfo->found_cpus);

    CPU_FOREACH(cpu) {

        set_bit(cpu->cpu_index, cpuinfo->found_cpus);

    }

}
