unsigned s390_del_running_cpu(S390CPU *cpu)

{

    CPUState *cs = CPU(cpu);



    if (cs->halted == 0) {

        assert(s390_running_cpus >= 1);

        s390_running_cpus--;

        cs->halted = 1;

        cs->exception_index = EXCP_HLT;

    }

    return s390_running_cpus;

}
