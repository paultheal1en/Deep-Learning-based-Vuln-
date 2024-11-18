void helper_lswx(CPUPPCState *env, target_ulong addr, uint32_t reg,

                 uint32_t ra, uint32_t rb)

{

    if (likely(xer_bc != 0)) {

        if (unlikely((ra != 0 && reg < ra && (reg + xer_bc) > ra) ||

                     (reg < rb && (reg + xer_bc) > rb))) {

            helper_raise_exception_err(env, POWERPC_EXCP_PROGRAM,

                                       POWERPC_EXCP_INVAL |

                                       POWERPC_EXCP_INVAL_LSWX);

        } else {

            helper_lsw(env, addr, xer_bc, reg);

        }

    }

}
