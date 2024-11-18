uint32_t HELPER(servc)(CPUS390XState *env, uint64_t r1, uint64_t r2)

{

    int r = sclp_service_call(env, r1, r2);

    if (r < 0) {

        program_interrupt(env, -r, 4);

        return 0;

    }

    return r;

}
