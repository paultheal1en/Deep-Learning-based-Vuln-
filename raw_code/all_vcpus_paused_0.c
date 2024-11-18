static int all_vcpus_paused(void)

{

    CPUState *penv = first_cpu;



    while (penv) {

        if (!penv->stopped)

            return 0;

        penv = (CPUState *)penv->next_cpu;

    }



    return 1;

}
