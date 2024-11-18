uint32_t HELPER(clcle)(CPUS390XState *env, uint32_t r1, uint64_t a2,

                       uint32_t r3)

{

    uintptr_t ra = GETPC();

    uint64_t destlen = get_length(env, r1 + 1);

    uint64_t dest = get_address(env, r1);

    uint64_t srclen = get_length(env, r3 + 1);

    uint64_t src = get_address(env, r3);

    uint8_t pad = a2 & 0xff;

    uint32_t cc = 0;



    if (!(destlen || srclen)) {

        return cc;

    }



    if (srclen > destlen) {

        srclen = destlen;

    }



    for (; destlen || srclen; src++, dest++, destlen--, srclen--) {

        uint8_t v1 = srclen ? cpu_ldub_data_ra(env, src, ra) : pad;

        uint8_t v2 = destlen ? cpu_ldub_data_ra(env, dest, ra) : pad;

        if (v1 != v2) {

            cc = (v1 < v2) ? 1 : 2;

            break;

        }

    }



    set_length(env, r1 + 1, destlen);

    /* can't use srclen here, we trunc'ed it */

    set_length(env, r3 + 1, env->regs[r3 + 1] - src - env->regs[r3]);

    set_address(env, r1, dest);

    set_address(env, r3, src);



    return cc;

}
