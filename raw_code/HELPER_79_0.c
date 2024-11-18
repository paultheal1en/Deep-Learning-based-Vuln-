uint32_t HELPER(mvcs)(CPUS390XState *env, uint64_t l, uint64_t a1, uint64_t a2)

{

    HELPER_LOG("%s: %16" PRIx64 " %16" PRIx64 " %16" PRIx64 "\n",

               __func__, l, a1, a2);



    return mvc_asc(env, l, a1, PSW_ASC_SECONDARY, a2, PSW_ASC_PRIMARY);

}
