static void kvm_inject_x86_mce_on(CPUState *env, struct kvm_x86_mce *mce,

                                  int flag)

{

    struct kvm_x86_mce_data data = {

        .env = env,

        .mce = mce,

        .abort_on_error = (flag & ABORT_ON_ERROR),

    };



    if (!env->mcg_cap) {

        fprintf(stderr, "MCE support is not enabled!\n");

        return;

    }



    run_on_cpu(env, kvm_do_inject_x86_mce, &data);

}
