static void par_write(CPUARMState *env, const ARMCPRegInfo *ri, uint64_t value)

{

    if (arm_feature(env, ARM_FEATURE_LPAE)) {

        env->cp15.par_el1 = value;

    } else if (arm_feature(env, ARM_FEATURE_V7)) {

        env->cp15.par_el1 = value & 0xfffff6ff;

    } else {

        env->cp15.par_el1 = value & 0xfffff1ff;

    }

}
