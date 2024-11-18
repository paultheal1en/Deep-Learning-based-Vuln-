target_ulong helper_load_slb_esid(CPUPPCState *env, target_ulong rb)

{

    target_ulong rt;



    if (ppc_load_slb_esid(env, rb, &rt) < 0) {

        helper_raise_exception_err(env, POWERPC_EXCP_PROGRAM,

                                   POWERPC_EXCP_INVAL);

    }

    return rt;

}
