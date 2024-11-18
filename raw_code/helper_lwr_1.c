target_ulong helper_lwr(CPUMIPSState *env, target_ulong arg1,

                        target_ulong arg2, int mem_idx)

{

    target_ulong tmp;



    tmp = do_lbu(env, arg2, mem_idx);

    arg1 = (arg1 & 0xFFFFFF00) | tmp;



    if (GET_LMASK(arg2) >= 1) {

        tmp = do_lbu(env, GET_OFFSET(arg2, -1), mem_idx);

        arg1 = (arg1 & 0xFFFF00FF) | (tmp << 8);

    }



    if (GET_LMASK(arg2) >= 2) {

        tmp = do_lbu(env, GET_OFFSET(arg2, -2), mem_idx);

        arg1 = (arg1 & 0xFF00FFFF) | (tmp << 16);

    }



    if (GET_LMASK(arg2) == 3) {

        tmp = do_lbu(env, GET_OFFSET(arg2, -3), mem_idx);

        arg1 = (arg1 & 0x00FFFFFF) | (tmp << 24);

    }

    return (int32_t)arg1;

}
