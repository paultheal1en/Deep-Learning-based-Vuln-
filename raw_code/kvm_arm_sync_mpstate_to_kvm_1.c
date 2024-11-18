int kvm_arm_sync_mpstate_to_kvm(ARMCPU *cpu)

{

    if (cap_has_mp_state) {

        struct kvm_mp_state mp_state = {

            .mp_state =

            cpu->powered_off ? KVM_MP_STATE_STOPPED : KVM_MP_STATE_RUNNABLE

        };

        int ret = kvm_vcpu_ioctl(CPU(cpu), KVM_SET_MP_STATE, &mp_state);

        if (ret) {

            fprintf(stderr, "%s: failed to set MP_STATE %d/%s\n",

                    __func__, ret, strerror(-ret));

            return -1;

        }

    }



    return 0;

}
