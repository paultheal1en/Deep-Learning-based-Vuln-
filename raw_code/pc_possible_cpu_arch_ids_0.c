static const CPUArchIdList *pc_possible_cpu_arch_ids(MachineState *machine)

{

    assert(machine->possible_cpus);

    return machine->possible_cpus;

}
