static void kvm_handle_internal_error(CPUState *env, struct kvm_run *run)

{



    if (kvm_check_extension(kvm_state, KVM_CAP_INTERNAL_ERROR_DATA)) {

        int i;



        fprintf(stderr, "KVM internal error. Suberror: %d\n",

                run->internal.suberror);



        for (i = 0; i < run->internal.ndata; ++i) {

            fprintf(stderr, "extra data[%d]: %"PRIx64"\n",

                    i, (uint64_t)run->internal.data[i]);

        }

    }

    cpu_dump_state(env, stderr, fprintf, 0);

    if (run->internal.suberror == KVM_INTERNAL_ERROR_EMULATION) {

        fprintf(stderr, "emulation failure\n");

        if (!kvm_arch_stop_on_emulation_error(env)) {

            return;

        }

    }

    /* FIXME: Should trigger a qmp message to let management know

     * something went wrong.

     */

    vm_stop(0);

}
