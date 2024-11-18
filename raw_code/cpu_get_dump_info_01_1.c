int cpu_get_dump_info(ArchDumpInfo *info,

                      const struct GuestPhysBlockList *guest_phys_blocks)

{

    PowerPCCPU *cpu = POWERPC_CPU(first_cpu);

    PowerPCCPUClass *pcc = POWERPC_CPU_GET_CLASS(cpu);



    info->d_machine = EM_PPC64;

    info->d_class = ELFCLASS64;

    if ((*pcc->interrupts_big_endian)(cpu)) {

        info->d_endian = ELFDATA2MSB;

    } else {

        info->d_endian = ELFDATA2LSB;









    return 0;
