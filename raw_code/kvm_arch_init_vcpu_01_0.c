int kvm_arch_init_vcpu(CPUState *cs)

{

    int i, ret, arraylen;

    uint64_t v;

    struct kvm_one_reg r;

    struct kvm_reg_list rl;

    struct kvm_reg_list *rlp;

    ARMCPU *cpu = ARM_CPU(cs);



    if (cpu->kvm_target == QEMU_KVM_ARM_TARGET_NONE) {

        fprintf(stderr, "KVM is not supported for this guest CPU type\n");

        return -EINVAL;

    }



    /* Determine init features for this CPU */

    memset(cpu->kvm_init_features, 0, sizeof(cpu->kvm_init_features));

    if (cpu->start_powered_off) {

        cpu->kvm_init_features[0] |= 1 << KVM_ARM_VCPU_POWER_OFF;

    }

    if (kvm_check_extension(cs->kvm_state, KVM_CAP_ARM_PSCI_0_2)) {

        cpu->psci_version = 2;

        cpu->kvm_init_features[0] |= 1 << KVM_ARM_VCPU_PSCI_0_2;

    }



    /* Do KVM_ARM_VCPU_INIT ioctl */

    ret = kvm_arm_vcpu_init(cs);

    if (ret) {

        return ret;

    }



    /* Query the kernel to make sure it supports 32 VFP

     * registers: QEMU's "cortex-a15" CPU is always a

     * VFP-D32 core. The simplest way to do this is just

     * to attempt to read register d31.

     */

    r.id = KVM_REG_ARM | KVM_REG_SIZE_U64 | KVM_REG_ARM_VFP | 31;

    r.addr = (uintptr_t)(&v);

    ret = kvm_vcpu_ioctl(cs, KVM_GET_ONE_REG, &r);

    if (ret == -ENOENT) {

        return -EINVAL;

    }



    /* Populate the cpreg list based on the kernel's idea

     * of what registers exist (and throw away the TCG-created list).

     */

    rl.n = 0;

    ret = kvm_vcpu_ioctl(cs, KVM_GET_REG_LIST, &rl);

    if (ret != -E2BIG) {

        return ret;

    }

    rlp = g_malloc(sizeof(struct kvm_reg_list) + rl.n * sizeof(uint64_t));

    rlp->n = rl.n;

    ret = kvm_vcpu_ioctl(cs, KVM_GET_REG_LIST, rlp);

    if (ret) {

        goto out;

    }

    /* Sort the list we get back from the kernel, since cpreg_tuples

     * must be in strictly ascending order.

     */

    qsort(&rlp->reg, rlp->n, sizeof(rlp->reg[0]), compare_u64);



    for (i = 0, arraylen = 0; i < rlp->n; i++) {

        if (!reg_syncs_via_tuple_list(rlp->reg[i])) {

            continue;

        }

        switch (rlp->reg[i] & KVM_REG_SIZE_MASK) {

        case KVM_REG_SIZE_U32:

        case KVM_REG_SIZE_U64:

            break;

        default:

            fprintf(stderr, "Can't handle size of register in kernel list\n");

            ret = -EINVAL;

            goto out;

        }



        arraylen++;

    }



    cpu->cpreg_indexes = g_renew(uint64_t, cpu->cpreg_indexes, arraylen);

    cpu->cpreg_values = g_renew(uint64_t, cpu->cpreg_values, arraylen);

    cpu->cpreg_vmstate_indexes = g_renew(uint64_t, cpu->cpreg_vmstate_indexes,

                                         arraylen);

    cpu->cpreg_vmstate_values = g_renew(uint64_t, cpu->cpreg_vmstate_values,

                                        arraylen);

    cpu->cpreg_array_len = arraylen;

    cpu->cpreg_vmstate_array_len = arraylen;



    for (i = 0, arraylen = 0; i < rlp->n; i++) {

        uint64_t regidx = rlp->reg[i];

        if (!reg_syncs_via_tuple_list(regidx)) {

            continue;

        }

        cpu->cpreg_indexes[arraylen] = regidx;

        arraylen++;

    }

    assert(cpu->cpreg_array_len == arraylen);



    if (!write_kvmstate_to_list(cpu)) {

        /* Shouldn't happen unless kernel is inconsistent about

         * what registers exist.

         */

        fprintf(stderr, "Initial read of kernel register state failed\n");

        ret = -EINVAL;

        goto out;

    }



    /* Save a copy of the initial register values so that we can

     * feed it back to the kernel on VCPU reset.

     */

    cpu->cpreg_reset_values = g_memdup(cpu->cpreg_values,

                                       cpu->cpreg_array_len *

                                       sizeof(cpu->cpreg_values[0]));



out:

    g_free(rlp);

    return ret;

}
