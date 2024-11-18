static int kvm_get_debugregs(CPUState *env)

{

#ifdef KVM_CAP_DEBUGREGS

    struct kvm_debugregs dbgregs;

    int i, ret;



    if (!kvm_has_debugregs()) {

        return 0;

    }



    ret = kvm_vcpu_ioctl(env, KVM_GET_DEBUGREGS, &dbgregs);

    if (ret < 0) {

       return ret;

    }

    for (i = 0; i < 4; i++) {

        env->dr[i] = dbgregs.db[i];

    }

    env->dr[4] = env->dr[6] = dbgregs.dr6;

    env->dr[5] = env->dr[7] = dbgregs.dr7;

#endif



    return 0;

}
