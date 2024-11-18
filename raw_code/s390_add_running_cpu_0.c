void s390_add_running_cpu(S390CPU *cpu)

{

    CPUState *cs = CPU(cpu);



    if (cs->halted) {

        s390_running_cpus++;

        cs->halted = 0;

        cs->exception_index = -1;

    }

}
