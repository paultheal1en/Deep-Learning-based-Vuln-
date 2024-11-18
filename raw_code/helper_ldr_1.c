target_ulong helper_ldr(CPUMIPSState *env, target_ulong arg1,

                        target_ulong arg2, int mem_idx)

{

    uint64_t tmp;



    tmp = do_lbu(env, arg2, mem_idx);

    arg1 = (arg1 & 0xFFFFFFFFFFFFFF00ULL) | tmp;



    if (GET_LMASK64(arg2) >= 1) {

        tmp = do_lbu(env, GET_OFFSET(arg2, -1), mem_idx);

        arg1 = (arg1 & 0xFFFFFFFFFFFF00FFULL) | (tmp  << 8);

    }



    if (GET_LMASK64(arg2) >= 2) {

        tmp = do_lbu(env, GET_OFFSET(arg2, -2), mem_idx);

        arg1 = (arg1 & 0xFFFFFFFFFF00FFFFULL) | (tmp << 16);

    }



    if (GET_LMASK64(arg2) >= 3) {

        tmp = do_lbu(env, GET_OFFSET(arg2, -3), mem_idx);

        arg1 = (arg1 & 0xFFFFFFFF00FFFFFFULL) | (tmp << 24);

    }



    if (GET_LMASK64(arg2) >= 4) {

        tmp = do_lbu(env, GET_OFFSET(arg2, -4), mem_idx);

        arg1 = (arg1 & 0xFFFFFF00FFFFFFFFULL) | (tmp << 32);

    }



    if (GET_LMASK64(arg2) >= 5) {

        tmp = do_lbu(env, GET_OFFSET(arg2, -5), mem_idx);

        arg1 = (arg1 & 0xFFFF00FFFFFFFFFFULL) | (tmp << 40);

    }



    if (GET_LMASK64(arg2) >= 6) {

        tmp = do_lbu(env, GET_OFFSET(arg2, -6), mem_idx);

        arg1 = (arg1 & 0xFF00FFFFFFFFFFFFULL) | (tmp << 48);

    }



    if (GET_LMASK64(arg2) == 7) {

        tmp = do_lbu(env, GET_OFFSET(arg2, -7), mem_idx);

        arg1 = (arg1 & 0x00FFFFFFFFFFFFFFULL) | (tmp << 56);

    }



    return arg1;

}
