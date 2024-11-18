void HELPER(access_check_cp_reg)(CPUARMState *env, void *rip)

{

    const ARMCPRegInfo *ri = rip;

    switch (ri->accessfn(env, ri)) {

    case CP_ACCESS_OK:

        return;

    case CP_ACCESS_TRAP:

    case CP_ACCESS_TRAP_UNCATEGORIZED:

        /* These cases will eventually need to generate different

         * syndrome information.

         */

        break;

    default:

        g_assert_not_reached();

    }

    raise_exception(env, EXCP_UDEF);

}
