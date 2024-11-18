static int kvm_sclp_service_call(CPUS390XState *env, struct kvm_run *run,

                                 uint16_t ipbh0)

{

    uint32_t sccb;

    uint64_t code;

    int r = 0;



    cpu_synchronize_state(env);

    sccb = env->regs[ipbh0 & 0xf];

    code = env->regs[(ipbh0 & 0xf0) >> 4];



    r = sclp_service_call(env, sccb, code);

    if (r) {

        setcc(env, 3);

    }



    return 0;

}
