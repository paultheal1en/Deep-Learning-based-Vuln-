static bool arm_cpu_has_work(CPUState *cs)

{

    ARMCPU *cpu = ARM_CPU(cs);



    return !cpu->powered_off

        && cs->interrupt_request &

        (CPU_INTERRUPT_FIQ | CPU_INTERRUPT_HARD

         | CPU_INTERRUPT_VFIQ | CPU_INTERRUPT_VIRQ

         | CPU_INTERRUPT_EXITTB);

}
