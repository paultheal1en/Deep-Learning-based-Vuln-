static bool pmsav7_needed(void *opaque)

{

    ARMCPU *cpu = opaque;

    CPUARMState *env = &cpu->env;



    return arm_feature(env, ARM_FEATURE_PMSA) &&

           arm_feature(env, ARM_FEATURE_V7);

}
