static int kvm_has_msr_star(CPUState *env)

{

    static int has_msr_star;

    int ret;



    /* first time */

    if (has_msr_star == 0) {        

        struct kvm_msr_list msr_list, *kvm_msr_list;



        has_msr_star = -1;



        /* Obtain MSR list from KVM.  These are the MSRs that we must

         * save/restore */

        msr_list.nmsrs = 0;

        ret = kvm_ioctl(env->kvm_state, KVM_GET_MSR_INDEX_LIST, &msr_list);

        if (ret < 0)

            return 0;



        /* Old kernel modules had a bug and could write beyond the provided

           memory. Allocate at least a safe amount of 1K. */

        kvm_msr_list = qemu_mallocz(MAX(1024, sizeof(msr_list) +

                                              msr_list.nmsrs *

                                              sizeof(msr_list.indices[0])));



        kvm_msr_list->nmsrs = msr_list.nmsrs;

        ret = kvm_ioctl(env->kvm_state, KVM_GET_MSR_INDEX_LIST, kvm_msr_list);

        if (ret >= 0) {

            int i;



            for (i = 0; i < kvm_msr_list->nmsrs; i++) {

                if (kvm_msr_list->indices[i] == MSR_STAR) {

                    has_msr_star = 1;

                    break;

                }

            }

        }



        free(kvm_msr_list);

    }



    if (has_msr_star == 1)

        return 1;

    return 0;

}
