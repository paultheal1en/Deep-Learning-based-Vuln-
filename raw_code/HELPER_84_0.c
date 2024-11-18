uint32_t HELPER(mvcle)(CPUS390XState *env, uint32_t r1, uint64_t a2,

                       uint32_t r3)

{

    uintptr_t ra = GETPC();

    uint64_t destlen = get_length(env, r1 + 1);

    uint64_t dest = get_address(env, r1);

    uint64_t srclen = get_length(env, r3 + 1);

    uint64_t src = get_address(env, r3);

    uint8_t pad = a2 & 0xff;

    uint8_t v;

    uint32_t cc;



    if (destlen == srclen) {

        cc = 0;

    } else if (destlen < srclen) {

        cc = 1;

    } else {

        cc = 2;

    }



    if (srclen > destlen) {

        srclen = destlen;

    }



    for (; destlen && srclen; src++, dest++, destlen--, srclen--) {

        v = cpu_ldub_data_ra(env, src, ra);

        cpu_stb_data_ra(env, dest, v, ra);

    }



    for (; destlen; dest++, destlen--) {

        cpu_stb_data_ra(env, dest, pad, ra);

    }



    set_length(env, r1 + 1 , destlen);

    /* can't use srclen here, we trunc'ed it */

    set_length(env, r3 + 1, env->regs[r3 + 1] - src - env->regs[r3]);

    set_address(env, r1, dest);

    set_address(env, r3, src);



    return cc;

}
