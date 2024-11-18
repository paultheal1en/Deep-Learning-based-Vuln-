static bool lowprot_enabled(const CPUS390XState *env)

{

    if (!(env->cregs[0] & CR0_LOWPROT)) {

        return false;

    }

    if (!(env->psw.mask & PSW_MASK_DAT)) {

        return true;

    }



    /* Check the private-space control bit */

    switch (env->psw.mask & PSW_MASK_ASC) {

    case PSW_ASC_PRIMARY:

        return !(env->cregs[1] & _ASCE_PRIVATE_SPACE);

    case PSW_ASC_SECONDARY:

        return !(env->cregs[7] & _ASCE_PRIVATE_SPACE);

    case PSW_ASC_HOME:

        return !(env->cregs[13] & _ASCE_PRIVATE_SPACE);

    default:

        /* We don't support access register mode */

        error_report("unsupported addressing mode");

        exit(1);

    }

}
