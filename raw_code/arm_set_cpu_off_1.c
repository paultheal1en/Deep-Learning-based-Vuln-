int arm_set_cpu_off(uint64_t cpuid)

{

    CPUState *target_cpu_state;

    ARMCPU *target_cpu;



    DPRINTF("cpu %" PRId64 "\n", cpuid);



    /* change to the cpu we are powering up */

    target_cpu_state = arm_get_cpu_by_id(cpuid);

    if (!target_cpu_state) {

        return QEMU_ARM_POWERCTL_INVALID_PARAM;

    }

    target_cpu = ARM_CPU(target_cpu_state);

    if (target_cpu->powered_off) {

        qemu_log_mask(LOG_GUEST_ERROR,

                      "[ARM]%s: CPU %" PRId64 " is already off\n",

                      __func__, cpuid);

        return QEMU_ARM_POWERCTL_IS_OFF;

    }



    target_cpu->powered_off = true;

    target_cpu_state->halted = 1;

    target_cpu_state->exception_index = EXCP_HLT;

    cpu_loop_exit(target_cpu_state);

    /* notreached */



    return QEMU_ARM_POWERCTL_RET_SUCCESS;

}
