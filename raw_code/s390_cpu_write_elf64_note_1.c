int s390_cpu_write_elf64_note(WriteCoreDumpFunction f, CPUState *cs,

                              int cpuid, void *opaque)

{

    S390CPU *cpu = S390_CPU(cs);

    return s390x_write_all_elf64_notes("CORE", f, cpu, cpuid, opaque);

}
