HELPER_LD_ATOMIC(ll, lw)

#ifdef TARGET_MIPS64

HELPER_LD_ATOMIC(lld, ld)

#endif

#undef HELPER_LD_ATOMIC



#define HELPER_ST_ATOMIC(name, ld_insn, st_insn, almask)                      \

target_ulong helper_##name(CPUMIPSState *env, target_ulong arg1,              \

                           target_ulong arg2, int mem_idx)                    \

{                                                                             \

    target_long tmp;                                                          \

                                                                              \

    if (arg2 & almask) {                                                      \

        env->CP0_BadVAddr = arg2;                                             \

        helper_raise_exception(env, EXCP_AdES);                               \

    }                                                                         \

    if (do_translate_address(env, arg2, 1) == env->lladdr) {                  \

        tmp = do_##ld_insn(env, arg2, mem_idx);                               \

        if (tmp == env->llval) {                                              \

            do_##st_insn(env, arg2, arg1, mem_idx);                           \

            return 1;                                                         \

        }                                                                     \

    }                                                                         \

    return 0;                                                                 \

}

HELPER_ST_ATOMIC(sc, lw, sw, 0x3)

#ifdef TARGET_MIPS64

HELPER_ST_ATOMIC(scd, ld, sd, 0x7)

#endif

#undef HELPER_ST_ATOMIC

#endif



#ifdef TARGET_WORDS_BIGENDIAN

#define GET_LMASK(v) ((v) & 3)

#define GET_OFFSET(addr, offset) (addr + (offset))

#else

#define GET_LMASK(v) (((v) & 3) ^ 3)

#define GET_OFFSET(addr, offset) (addr - (offset))

#endif



target_ulong helper_lwl(CPUMIPSState *env, target_ulong arg1,

                        target_ulong arg2, int mem_idx)

{

    target_ulong tmp;



    tmp = do_lbu(env, arg2, mem_idx);

    arg1 = (arg1 & 0x00FFFFFF) | (tmp << 24);



    if (GET_LMASK(arg2) <= 2) {

        tmp = do_lbu(env, GET_OFFSET(arg2, 1), mem_idx);

        arg1 = (arg1 & 0xFF00FFFF) | (tmp << 16);

    }



    if (GET_LMASK(arg2) <= 1) {

        tmp = do_lbu(env, GET_OFFSET(arg2, 2), mem_idx);

        arg1 = (arg1 & 0xFFFF00FF) | (tmp << 8);

    }



    if (GET_LMASK(arg2) == 0) {

        tmp = do_lbu(env, GET_OFFSET(arg2, 3), mem_idx);

        arg1 = (arg1 & 0xFFFFFF00) | tmp;

    }

    return (int32_t)arg1;

}
