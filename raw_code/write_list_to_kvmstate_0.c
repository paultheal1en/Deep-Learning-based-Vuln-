bool write_list_to_kvmstate(ARMCPU *cpu)

{

    CPUState *cs = CPU(cpu);

    int i;

    bool ok = true;



    for (i = 0; i < cpu->cpreg_array_len; i++) {

        struct kvm_one_reg r;

        uint64_t regidx = cpu->cpreg_indexes[i];

        uint32_t v32;

        int ret;



        r.id = regidx;

        switch (regidx & KVM_REG_SIZE_MASK) {

        case KVM_REG_SIZE_U32:

            v32 = cpu->cpreg_values[i];

            r.addr = (uintptr_t)&v32;

            break;

        case KVM_REG_SIZE_U64:

            r.addr = (uintptr_t)(cpu->cpreg_values + i);

            break;

        default:

            abort();

        }

        ret = kvm_vcpu_ioctl(cs, KVM_SET_ONE_REG, &r);

        if (ret) {

            /* We might fail for "unknown register" and also for

             * "you tried to set a register which is constant with

             * a different value from what it actually contains".

             */

            ok = false;

        }

    }

    return ok;

}
